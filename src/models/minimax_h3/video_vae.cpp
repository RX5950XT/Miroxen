#include "models/minimax_h3/video_vae.h"
#include "models/minimax_h3/video_vae_kernels.h"
#include "core/dequant.h"
#include "core/weights.h"
#include "kernels/attention.h"
#include "kernels/ops.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

namespace mx {
namespace {
// 只用於讀取這個模型的設定，不依賴外部 JSON 套件。
struct ConfigValue {
  std::map<std::string, ConfigValue> object;
  std::vector<ConfigValue> array;
  std::string text;
  char kind = 0;
  const ConfigValue& at(const std::string& name) const {
    auto it=object.find(name);
    MX_CHECK(kind=='{' && it!=object.end(), "H3 VAE 設定缺少 %s", name.c_str());
    return it->second;
  }
  double number() const {
    MX_CHECK(kind=='n', "H3 VAE 設定需要數字");
    size_t end=0; double v=std::stod(text,&end);
    MX_CHECK(end==text.size() && std::isfinite(v), "H3 VAE 設定數字無效"); return v;
  }
  int integer() const {
    double v=number(); MX_CHECK(v>=0 && v<=4096 && v==std::floor(v), "H3 VAE 設定整數超出範圍"); return int(v);
  }
};
class ConfigParser {
 public:
  explicit ConfigParser(std::string text): text_(std::move(text)) {}
  ConfigValue parse() {auto v=value(0); space(); MX_CHECK(pos_==text_.size(), "H3 VAE JSON 結尾無效"); return v;}
 private:
  std::string text_; size_t pos_=0;
  void space(){while(pos_<text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;}
  char peek(){space(); MX_CHECK(pos_<text_.size(),"H3 VAE JSON 截斷"); return text_[pos_];}
  void take(char c){MX_CHECK(peek()==c,"H3 VAE JSON 格式錯誤，位置 %zu",pos_); ++pos_;}
  std::string string(){
    take('"'); std::string result;
    while(pos_<text_.size()) {
      char c=text_[pos_++]; if(c=='"') return result;
      MX_CHECK(static_cast<unsigned char>(c)>=32,"H3 VAE JSON 字串無效");
      if(c=='\\') {
        MX_CHECK(pos_<text_.size(),"H3 VAE JSON escape 截斷"); c=text_[pos_++];
        if(c=='n') c='\n'; else if(c=='r') c='\r'; else if(c=='t') c='\t';
        else MX_CHECK(c=='"'||c=='\\'||c=='/',"H3 VAE 設定不支援此 escape");
      }
      result+=c;
    }
    throw std::runtime_error("H3 VAE JSON 字串截斷");
  }
  ConfigValue value(int depth){
    MX_CHECK(depth<32,"H3 VAE JSON 巢狀過深"); ConfigValue v; v.kind=peek();
    if(v.kind=='"'){v.text=string();return v;}
    if(v.kind=='{'||v.kind=='[') {
      char close=v.kind=='{'?'}':']'; ++pos_;
      if(peek()==close){++pos_;return v;}
      for(;;){
        if(v.kind=='{'){auto key=string();take(':'); MX_CHECK(v.object.emplace(key,value(depth+1)).second,"H3 VAE JSON 重複鍵");}
        else v.array.push_back(value(depth+1));
        char next=peek(); ++pos_; if(next==close) return v;
        MX_CHECK(next==',',"H3 VAE JSON 分隔符無效");
      }
    }
    size_t start=pos_;
    while(pos_<text_.size() && text_[pos_]!=',' && text_[pos_]!='}' && text_[pos_]!=']' &&
        !std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    v.text=text_.substr(start,pos_-start); v.kind='n';
    if(v.text=="true"||v.text=="false"||v.text=="null") v.kind='l'; else v.number();
    return v;
  }
};
ConfigValue read_config() {
  const char* env=std::getenv("MX_H3_VIDEO_VAE_CONFIG");
  std::vector<std::filesystem::path> paths;
  if(env && *env) paths.emplace_back(env);
  else {
    paths.emplace_back("models/minimax-h3/hf/vae/config.json");
    paths.emplace_back("../../Miroxen/models/minimax-h3/hf/vae/config.json");
  }
  for(const auto& path:paths) {
    if(!std::filesystem::exists(path)) continue;
    MX_CHECK(std::filesystem::file_size(path)<=65536,"H3 VAE 設定過大");
    std::ifstream f(path,std::ios::binary); MX_CHECK(f.good(),"無法讀取 H3 VAE config.json");
    std::ostringstream s;s<<f.rdbuf();MX_CHECK(!f.bad(),"H3 VAE 設定讀取失敗");
    return ConfigParser(s.str()).parse();
  }
  throw std::runtime_error("找不到 H3 VAE config.json；請設定 MX_H3_VIDEO_VAE_CONFIG");
}
std::vector<int> int_array(const ConfigValue& v) {
  MX_CHECK(v.kind=='[' && !v.array.empty(),"H3 VAE 設定需要非空陣列");
  std::vector<int> a; for(const auto& x:v.array) a.push_back(x.integer()); return a;
}
std::vector<float> float_array(const ConfigValue& v) {
  MX_CHECK(v.kind=='[' && v.array.size()==24,"H3 VAE latent 統計必須有 24 個值");
  std::vector<float> a;for(const auto& x:v.array) {
    float n=float(x.number());MX_CHECK(std::isfinite(n),"H3 VAE latent 統計超出 f32 範圍");a.push_back(n);
  }
  return a;
}
struct Tiles {std::vector<int> starts,lengths,overlaps;};
Tiles split_tiles(int length,int tile,int overlap,int ratio) {
  if(length<=tile) return {{0},{length},{}};
  int n=(length+tile-1)/tile;
  while(tile*n-overlap*(n-1)<length) ++n;
  Tiles t; t.overlaps.assign(n-1,overlap); t.lengths.assign(n,tile);
  int slack=tile*n-overlap*(n-1)-length;
  MX_CHECK(slack%ratio==0,"H3 VAE tile 未對齊 latent");
  for(int i=0;i<slack/ratio;++i) t.overlaps[i%(n-1)]+=ratio;
  t.starts.push_back(0);
  for(int o:t.overlaps) t.starts.push_back(t.starts.back()+tile-o);
  return t;
}
// 和 diffusers _stitch_tiles 一樣，鄰接來源是尚未混合的原始 tile。
void stitch(const std::vector<float>& tile,const std::vector<float>* above,const std::vector<float>* left,
    std::vector<float>& out,int frames,int channels,int H,int W,int h,int w,int y,int x,
    int oh,int ow,int keep_h,int keep_w) {
  for(int t=0;t<frames;++t) for(int yy=0;yy<keep_h;++yy) for(int xx=0;xx<keep_w;++xx)
    for(int c=0;c<channels;++c) {
      size_t i=((size_t(t)*h+yy)*w+xx)*channels+c; float v=tile[i];
      if(above && yy<oh) {
        float a=(*above)[((size_t(t)*h+h-oh+yy)*w+xx)*channels+c];
        float f=float(yy)/oh; v=a*(1.f-f)+v*f;
      }
      if(left && xx<ow) {
        float a=(*left)[((size_t(t)*h+yy)*w+w-ow+xx)*channels+c];
        float f=float(xx)/ow; v=a*(1.f-f)+v*f;
      }
      out[((size_t(t)*H+y+yy)*W+x+xx)*channels+c]=v;
    }
}
// l：最後一個時間平面（單幀路徑只看得到它）；taps：全部時間平面（多幀因果 3D 卷積，前面補 kt-1 個零幀），
// 第一次編碼影片時才載入（src 指向原始權重）。
struct Conv {Linear l;std::vector<Linear> taps;const WeightRef* src=nullptr;int kh=1,kw=1,kt=1,stride=1,pad=0,tstride=1;};
struct Resnet {Conv c1,c2,shortcut;const float *w1=nullptr,*b1=nullptr,*w2=nullptr,*b2=nullptr;};
struct Level {std::vector<Resnet> blocks;Conv down;};
struct Block {Linear q,k,v,qkv,proj,ff1,ff2;const float *n1,*n2,*s1,*s2;};
// 配置順序與真正的 kernel 路徑一致，含 Arena 的 256-byte 對齊。
struct Budget {
  size_t used=0,peak=0;
  void alloc(size_t n){used=(used+255)&~size_t(255); used+=n;peak=std::max(peak,used);}
};
}  // namespace

struct H3VideoVae::Impl {
  DeviceWeights small;
  std::map<std::string,DevBuf> matrices;
  std::vector<Block> blocks;
  std::vector<Level> levels;
  Linear post,proj_in,proj_out;
  Conv conv_in,conv_out,quant;
  const float *registers,*out_w,*out_b,*enc_w,*enc_b;
  DevBuf mean_buf,std_buf,ones_buf,norm_scale;
  int device,D,heads,head_dim,nreg,ff,rot,p,pt,clip,drop,chunk,overlap,pre,foverlap,groups;
  float eps,enc_eps,theta;
  bool comfy;
  std::vector<int> channels,space_down,time_down;

  Impl(const TensorStore& st,int dev):small(st,dev),device(dev) {
    MX_CUDA(cudaSetDevice(dev)); const auto cfg=read_config();
    auto num=[&](const char* n){return cfg.at(n).integer();};
    MX_CHECK(num("in_channels")==3 && num("out_channels")==3 && num("latent_channels")==24,
        "H3 VAE 通道與介面不符");
    channels=int_array(cfg.at("block_out_channels")); space_down=int_array(cfg.at("spatial_downsample_factors"));
    time_down=int_array(cfg.at("temporal_downsample_factors"));
    MX_CHECK(channels.size()==space_down.size() && channels.size()==time_down.size() && channels.size()<=16,"H3 VAE encoder 設定不符");
    p=pt=1; for(size_t i=0;i<channels.size();++i) {
      MX_CHECK(channels[i]>0 && channels[i]<=2048 && (space_down[i]==1||space_down[i]==2) &&
          (time_down[i]==1||time_down[i]==2),"H3 VAE encoder 層設定無效");
      p*=space_down[i];pt*=time_down[i];
    }
    MX_CHECK(p==16 && pt==4,"H3 VAE 壓縮比例與介面不符");
    heads=num("decoder_num_attention_heads");head_dim=num("decoder_attention_head_dim");
    MX_CHECK(heads==32 && head_dim==64 && num("decoder_ffn_mult")==4,"不支援此 H3 VAE attention 設定");
    D=heads*head_dim;nreg=num("decoder_num_register_tokens");ff=num("decoder_ffn_mult")*D;
    eps=float(cfg.at("decoder_norm_eps").number());enc_eps=float(cfg.at("norm_eps").number());
    double ratio=cfg.at("decoder_rope_dim_ratio").number();
    MX_CHECK(ratio>0 && ratio<=1,"H3 VAE RoPE 比例無效");
    theta=float(cfg.at("decoder_rope_theta").number());rot=int(head_dim*ratio);
    groups=num("norm_num_groups");clip=num("clip_length");drop=num("token_drop");
    MX_CHECK(nreg==4 && rot==48 && head_dim*ratio==rot && theta>0 && std::isfinite(theta) &&
        eps>0 && std::isfinite(eps) && enc_eps>0 && std::isfinite(enc_eps),
        "不支援此 H3 VAE decoder 設定");
    MX_CHECK(groups>0 && clip==17 && drop==3 && cfg.at("spatial_padding_mode").text=="reflect","H3 VAE chunk 或 padding 設定不符");
    pre=(pt-clip%pt)%pt;chunk=(clip+pt-1)/pt;overlap=(chunk-drop%chunk)%chunk;foverlap=std::max(overlap*pt-pre,0);
    const auto mean=float_array(cfg.at("latents_mean")),std=float_array(cfg.at("latents_std"));
    for(float v:std) MX_CHECK(v>0 && std::isfinite(v),"H3 VAE latent std 必須大於零");
    mean_buf=upload(mean);std_buf=upload(std);ones_buf=upload(std::vector<float>(head_dim,1.f));
    comfy=st.has("decoder.x_embedder.weight");
    post=load_linear("post_quant_conv",24,24,true);
    proj_in=load_linear(comfy?"decoder.x_embedder":"decoder.proj_in",24,D);
    proj_out=load_linear("decoder.proj_out",D,3*pt*p*p);
    registers=f32("decoder.register_tokens",{1,nreg,D});
    out_w=f32("decoder.norm_out.weight",{D});out_b=f32("decoder.norm_out.bias",{D});
    std::vector<float> scale(D);
    const auto& nw=st.get("decoder.norm_out.weight");dequant_cpu(nw.data,nw.type,D,scale.data());
    for(auto& value:scale)value-=1.f;norm_scale=upload(scale);
    int layers=num("decoder_num_layers"); MX_CHECK(layers==36,"H3 VAE decoder 層數不符");
    for(int i=0;i<layers;++i) load_block(i);
    load_encoder(num("layers_per_block"));
    MX_INFO("H3 影片 VAE：%s，權重 %.1f MiB",comfy?"Comfy":"diffusers",weight_bytes()/1048576.0);
  }
  DevBuf upload(const std::vector<float>& data) {
    DevBuf b(data.size()*sizeof(float),device);MX_CUDA(cudaMemcpy(b.get(),data.data(),b.size(),cudaMemcpyHostToDevice));return b;
  }
  const float* f32(const std::string& name,const std::vector<int64_t>& shape) {
    MX_CHECK(small.store().get(name).shape==shape,"H3 VAE 張量形狀不符：%s",name.c_str());return small.f32(name);
  }
  Linear load_linear(const std::string& name,int in,int out,bool conv=false) {
    const auto& r=small.store().get(name+".weight");
    std::vector<int64_t> expected=conv?std::vector<int64_t>{out,in,1,1,1}:std::vector<int64_t>{out,in};
    MX_CHECK(r.shape==expected && (r.type==GType::F16||r.type==GType::BF16||r.type==GType::F32),"H3 VAE 權重形狀／型別錯誤：%s",name.c_str());
    DevBuf raw(r.nbytes,device);MX_CUDA(cudaMemcpy(raw.get(),r.data,r.nbytes,cudaMemcpyHostToDevice));
    DevBuf converted(size_t(in)*out*sizeof(bf16),device);
    dequant_to_bf16(raw.get(),r.type,r.numel(),converted.as<bf16>(),nullptr);MX_CUDA(cudaDeviceSynchronize());
    if(name.ends_with(".attn.to_qkv")) {
      // Comfy 是 [head, Q/K/V, dim]，共用 attention 需要 [Q/K/V, head, dim]。
      DevBuf ordered(converted.size(),device);
      for(int h=0;h<heads;++h) for(int q=0;q<3;++q) {
        size_t src=size_t(h*3+q)*head_dim*in,dst=size_t(q*heads+h)*head_dim*in;
        MX_CUDA(cudaMemcpy(ordered.as<bf16>()+dst,converted.as<bf16>()+src,
            size_t(head_dim)*in*sizeof(bf16),cudaMemcpyDeviceToDevice));
      }
      converted=std::move(ordered);
    }
    Linear l;l.name=name;l.in=in;l.out=out;l.w={converted.get(),GType::BF16,{out,in}};
    l.bias=f32(name+".bias",{out});
    if(name.ends_with(".attn.to_qkv")) {
      const auto& b=small.store().get(name+".bias");std::vector<float> source(out),ordered(out);
      dequant_cpu(b.data,b.type,out,source.data());
      for(int h=0;h<heads;++h) for(int q=0;q<3;++q)
        std::copy_n(source.data()+(h*3+q)*head_dim,head_dim,ordered.data()+(q*heads+h)*head_dim);
      auto gpu=upload(ordered);l.bias=gpu.as<float>();matrices.emplace(name+".bias",std::move(gpu));
    }
    matrices.emplace(name,std::move(converted));return l;
  }
  Conv load_conv(const std::string& name,int in,int out,int stride=1,int pad=1) {
    const auto& r=small.store().get(name+".weight");
    MX_CHECK(r.shape==std::vector<int64_t>({out,in,3,3,3}) || r.shape==std::vector<int64_t>({out,in,1,1,1}),"H3 VAE conv 形狀錯誤：%s",name.c_str());
    MX_CHECK(r.type==GType::F16||r.type==GType::BF16||r.type==GType::F32,"H3 VAE conv 型別錯誤");
    Conv cv;cv.kh=int(r.shape[3]);cv.kw=int(r.shape[4]);cv.kt=int(r.shape[2]);cv.stride=stride;cv.pad=cv.kh==1?0:pad;
    cv.src=&r;cv.l=conv_plane(cv,cv.kt-1,f32(name+".bias",{out}));return cv;
  }
  // 第 k 個時間平面 → [out, in·kh·kw] 的 bf16 線性層。
  Linear conv_plane(const Conv& cv,int k,const float* bias) {
    const WeightRef& r=*cv.src;
    const int out=int(r.shape[0]),in=int(r.shape[1]),hw=cv.kh*cv.kw,K=in*hw;
    std::vector<float> host(size_t(out)*K);
    for(int o=0;o<out;++o) for(int c=0;c<in;++c) {
      size_t src=((size_t(o)*in+c)*cv.kt+k)*hw;
      dequant_cpu(static_cast<const char*>(r.data)+src*gtype_block_bytes(r.type),r.type,hw,host.data()+size_t(o)*K+c*hw);
    }
    DevBuf raw=upload(host),converted(size_t(out)*K*sizeof(bf16),device);
    dequant_to_bf16(raw.get(),GType::F32,int64_t(out)*K,converted.as<bf16>(),nullptr);MX_CUDA(cudaDeviceSynchronize());
    Linear l;l.name=r.name;l.in=K;l.out=out;l.w={converted.get(),GType::BF16,{out,K}};l.bias=bias;
    matrices.emplace(r.name+"#"+std::to_string(k),std::move(converted));return l;
  }
  void ensure_taps(Conv& cv) {
    if(!cv.src || !cv.taps.empty()) return;
    for(int k=0;k<cv.kt-1;++k) cv.taps.push_back(conv_plane(cv,k,cv.l.bias));
    cv.taps.push_back(cv.l);
  }
  void ensure_all_taps() {
    ensure_taps(conv_in);ensure_taps(conv_out);ensure_taps(quant);
    for(auto& lv:levels) {
      for(auto& r:lv.blocks) {ensure_taps(r.c1);ensure_taps(r.c2);ensure_taps(r.shortcut);}
      ensure_taps(lv.down);
    }
  }
  void load_block(int i) {
    auto base="decoder.transformer_blocks."+std::to_string(i);Block b;
    b.n1=f32(base+".norm1.weight",{D});b.n2=f32(base+".norm2.weight",{D});
    b.s1=f32(base+".scale1",{D});b.s2=f32(base+".scale2",{D});
    if(comfy) b.qkv=load_linear(base+".attn.to_qkv",D,3*D);
    else {
      b.q=load_linear(base+".attn.to_q",D,D);b.k=load_linear(base+".attn.to_k",D,D);b.v=load_linear(base+".attn.to_v",D,D);
    }
    b.proj=load_linear(base+(comfy?".attn.to_out":".attn.to_out.0"),D,D);
    b.ff1=load_linear(base+(comfy?".ff.w1":".ff.net.0.proj"),D,2*ff);
    b.ff2=load_linear(base+(comfy?".ff.w2":".ff.net.2"),ff,D);blocks.push_back(std::move(b));
  }
  void load_encoder(int count) {
    MX_CHECK(count>0 && count<=16,"H3 VAE encoder block 數無效");
    conv_in=load_conv("encoder.conv_in",3,channels[0]);int in=channels[0];
    for(size_t i=0;i<channels.size();++i) {
      MX_CHECK(channels[i]%groups==0 && in%groups==0,"H3 VAE GroupNorm 通道不符");Level level;
      for(int j=0;j<count;++j) {
        auto n=comfy?"encoder.down."+std::to_string(i)+".block."+std::to_string(j):
            "encoder.down_blocks."+std::to_string(i)+".resnets."+std::to_string(j);
        Resnet r;r.w1=f32(n+".norm1.weight",{in});r.b1=f32(n+".norm1.bias",{in});
        r.w2=f32(n+".norm2.weight",{channels[i]});r.b2=f32(n+".norm2.bias",{channels[i]});
        r.c1=load_conv(n+".conv1",in,channels[i]);r.c2=load_conv(n+".conv2",channels[i],channels[i]);
        if(in!=channels[i]) r.shortcut=load_conv(n+(comfy?".nin_shortcut":".conv_shortcut"),in,channels[i]);
        level.blocks.push_back(std::move(r));in=channels[i];
      }
      if(space_down[i]*time_down[i]>1) {
        auto n=comfy?"encoder.down."+std::to_string(i)+".downsample.conv":
            "encoder.down_blocks."+std::to_string(i)+".downsamplers.0.conv";
        level.down=load_conv(n,in,in,space_down[i],0);level.down.tstride=time_down[i];
      }
      levels.push_back(std::move(level));
    }
    enc_w=f32("encoder.norm_out.weight",{in});enc_b=f32("encoder.norm_out.bias",{in});
    conv_out=load_conv("encoder.conv_out",in,48);quant=load_conv("quant_conv",48,48);
  }
  size_t weight_bytes() const {size_t n=small.bytes();for(auto& [name,b]:matrices)n+=b.size();return n;}

  void transformer(Ctx& c,const Block& b,float* x,int S,const float* cs,const float* sn) {
    {
      ArenaScope scope(c.act); auto norm=c.act.alloc_n<bf16>(size_t(S)*D);
      rmsnorm_mod(x,S,D,eps,b.n1,{}, {},norm,c.stream);
      auto qkv=c.act.alloc_n<bf16>(size_t(S)*3*D);
      if(comfy) linear(c,b.qkv,norm,S,qkv);
      else {linear(c,b.q,norm,S,qkv);linear(c,b.k,norm,S,qkv+size_t(S)*D);linear(c,b.v,norm,S,qkv+size_t(S)*2*D);}
      auto q=c.act.alloc_n<bf16>(size_t(S)*D),k=c.act.alloc_n<bf16>(size_t(S)*D),v=c.act.alloc_n<bf16>(size_t(S)*D);
      int stride=comfy?3*D:D;
      head_norm_rope(qkv,S,stride,0,heads,head_dim,ones_buf.as<float>(),eps,cs,sn,rot,ROPE_HALF,q,c.stream);
      head_norm_rope(comfy?qkv:qkv+size_t(S)*D,S,stride,comfy?D:0,heads,head_dim,ones_buf.as<float>(),eps,cs,sn,rot,ROPE_HALF,k,c.stream);
      head_norm_rope(comfy?qkv:qkv+size_t(S)*2*D,S,stride,comfy?2*D:0,heads,head_dim,nullptr,eps,nullptr,nullptr,0,ROPE_NONE,v,c.stream);
      auto a=c.act.alloc_n<bf16>(size_t(S)*D),out=c.act.alloc_n<bf16>(size_t(S)*D);
      AttnParams params;params.Sq=params.Skv=S;params.Hq=params.Hkv=heads;params.D=head_dim;
      flash_attn(q,k,v,a,params,c.stream);linear(c,b.proj,a,S,out);
      gated_residual(x,out,S,D,{b.s1},0,c.stream);
    }
    {
      ArenaScope scope(c.act);auto norm=c.act.alloc_n<bf16>(size_t(S)*D);
      rmsnorm_mod(x,S,D,eps,b.n2,{}, {},norm,c.stream);
      auto gu=c.act.alloc_n<bf16>(size_t(S)*2*ff),hidden=c.act.alloc_n<bf16>(size_t(S)*ff);
      auto out=c.act.alloc_n<bf16>(size_t(S)*D);
      linear(c,b.ff1,norm,S,gu);h3v::swiglu(gu,hidden,S,ff,comfy,c.stream);linear(c,b.ff2,hidden,S,out);
      gated_residual(x,out,S,D,{b.s2},0,c.stream);
    }
  }
  std::vector<float> decode_tile(Ctx& c,const float* z,int F,int H,int W,int start,int T,
      int y,int x,int h,int w) {
    ArenaScope scope(c.act);int N=T*h*w,S=N+nreg+1;
    auto states=c.act.alloc_n<float>(size_t(S)*D);
    auto cs=c.act.alloc_n<float>(size_t(S)*rot/2),sn=c.act.alloc_n<float>(size_t(S)*rot/2);
    h3v::rope(cs,sn,T,h,w,nreg+1,rot,theta,c.stream);
    {
      ArenaScope init(c.act);auto input=c.act.alloc_n<bf16>(size_t(N)*24),post_z=c.act.alloc_n<bf16>(size_t(N)*24);
      auto projected=c.act.alloc_n<bf16>(size_t(N)*D);
      h3v::gather_latent(z,input,F,H,W,start,T,y,x,h,w,mean_buf.as<float>(),std_buf.as<float>(),c.stream);
      linear(c,post,input,N,post_z);linear(c,proj_in,post_z,N,projected);
      h3v::anchors(projected,states,registers,N,D,nreg,c.stream);
    }
    for(const auto& b:blocks) transformer(c,b,states,S,cs,sn);
    auto norm=c.act.alloc_n<bf16>(size_t(N)*D);
    layernorm_mod(states,N,D,eps,{norm_scale.as<float>()}, {out_b},norm,c.stream);
    auto projected=c.act.alloc_n<bf16>(size_t(N)*proj_out.out);
    linear(c,proj_out,norm,N,projected);
    auto pixels=c.act.alloc_n<float>(size_t(T)*pt*h*p*w*p*3);
    h3v::unpack(projected,pixels,T,h,w,pt,p,c.stream);
    std::vector<float> host(size_t(T)*pt*h*p*w*p*3);
    MX_CUDA(cudaMemcpyAsync(host.data(),pixels,host.size()*sizeof(float),cudaMemcpyDeviceToHost,c.stream));c.sync();return host;
  }
  std::vector<float> decode_clip(Ctx& c,const float* z,int F,int H,int W,int start,int T) {
    auto ys=split_tiles(H*p,256,64,p),xs=split_tiles(W*p,256,64,p);
    std::vector<float> result(size_t(T)*pt*H*p*W*p*3);
    std::vector<std::vector<float>> previous;
    for(size_t i=0;i<ys.starts.size();++i) {
      std::vector<std::vector<float>> row;
      for(size_t j=0;j<xs.starts.size();++j) {
        row.push_back(decode_tile(c,z,F,H,W,start,T,ys.starts[i]/p,xs.starts[j]/p,ys.lengths[i]/p,xs.lengths[j]/p));
        int h=ys.lengths[i],w=xs.lengths[j],oh=i?ys.overlaps[i-1]:0,ow=j?xs.overlaps[j-1]:0;
        int kh=h-(i+1<ys.starts.size()?ys.overlaps[i]:0),kw=w-(j+1<xs.starts.size()?xs.overlaps[j]:0);
        stitch(row.back(),i?&previous[j]:nullptr,j?&row[j-1]:nullptr,result,T*pt,3,H*p,W*p,
            h,w,ys.starts[i],xs.starts[j],oh,ow,kh,kw);
      }
      previous=std::move(row);
    }
    return result;
  }
  void conv(Ctx& c,const Conv& cv,const bf16* input,bf16* output,int H,int W,int C,int OH,int OW) {
    ArenaScope scope(c.act);int K=C*cv.kh*cv.kw,N=OH*OW,rows=std::min(N,4096);
    auto col=c.act.alloc_n<bf16>(size_t(rows)*K);
    for(int start=0;start<N;start+=rows) {
      int n=std::min(rows,N-start);
      h3v::im2col(input,col,H,W,C,cv.kh,cv.kw,cv.stride,cv.pad,OH,OW,start,n,c.stream);
      linear(c,cv.l,col,n,output+size_t(start)*cv.l.out);
    }
  }
  void resnet(Ctx& c,const Resnet& r,const bf16* input,bf16* output,int H,int W,int C) {
    ArenaScope scope(c.act);int N=H*W,OC=int(r.c1.l.out);
    auto norm1=c.act.alloc_n<bf16>(size_t(N)*C),hidden=c.act.alloc_n<bf16>(size_t(N)*OC);
    auto norm2=c.act.alloc_n<bf16>(size_t(N)*OC);
    h3v::groupnorm_silu(input,norm1,r.w1,r.b1,N,C,groups,enc_eps,c.stream);
    conv(c,r.c1,norm1,hidden,H,W,C,H,W);
    h3v::groupnorm_silu(hidden,norm2,r.w2,r.b2,N,OC,groups,enc_eps,c.stream);
    conv(c,r.c2,norm2,output,H,W,OC,H,W);
    if(r.shortcut.l) {conv(c,r.shortcut,input,hidden,H,W,C,H,W);add_inplace_bf16(output,hidden,size_t(N)*OC,c.stream);}
    else add_inplace_bf16(output,input,size_t(N)*OC,c.stream);
  }
  std::vector<float> encode_tile(Ctx& c,const uint8_t* rgb,int H,int W,int y,int x,int h,int w) {
    ArenaScope scope(c.act);int max_c=*std::max_element(channels.begin(),channels.end());
    // 依實際設定算兩個 ping-pong 緩衝區，包含每層變更通道前後的大小。
    size_t capacity=size_t(h)*w*channels[0];int bh=h,bw=w;
    for(size_t i=0;i<channels.size();++i) {
      capacity=std::max(capacity,size_t(bh)*bw*channels[i]);
      if(levels[i].down.l) {bh=space_down[i]==2?bh/2:bh-2;bw=space_down[i]==2?bw/2:bw-2;}
    }
    capacity=std::max(capacity,size_t(h/p)*w/p*max_c);
    auto a=c.act.alloc_n<bf16>(capacity),b=c.act.alloc_n<bf16>(capacity);
    auto input=c.act.alloc_n<bf16>(size_t(h)*w*3);
    h3v::image_input(rgb,input,H,W,y,x,h,w,c.stream);conv(c,conv_in,input,a,h,w,3,h,w);
    int C=channels[0],hh=h,ww=w;
    for(size_t i=0;i<levels.size();++i) {
      for(const auto& r:levels[i].blocks) {resnet(c,r,a,b,hh,ww,C);std::swap(a,b);C=int(r.c1.l.out);}
      if(levels[i].down.l) {
        int oh=space_down[i]==2?hh/2:hh-2,ow=space_down[i]==2?ww/2:ww-2;
        MX_CHECK(oh>0 && ow>0,"H3 VAE encoder 空間大小不足");
        conv(c,levels[i].down,a,b,hh,ww,C,oh,ow);std::swap(a,b);hh=oh;ww=ow;
      }
    }
    h3v::groupnorm_silu(a,b,enc_w,enc_b,hh*ww,C,groups,enc_eps,c.stream);
    conv(c,conv_out,b,a,hh,ww,C,hh,ww);conv(c,quant,a,b,hh,ww,48,hh,ww);
    auto moments=c.act.alloc_n<float>(size_t(hh)*ww*48);bf16_to_f32(b,moments,size_t(hh)*ww*48,c.stream);
    std::vector<float> host(size_t(hh)*ww*48);
    MX_CUDA(cudaMemcpyAsync(host.data(),moments,host.size()*sizeof(float),cudaMemcpyDeviceToHost,c.stream));c.sync();return host;
  }
  // 因果 3D 卷積：in 為 Tin 個連續的 [H, W, C] 幀；第 t 個輸出 = Σ_k taps[k]·im2col(in[t·ts + k − (kt−1)])，
  // 索引為負的幀是補零（直接略過）。回傳輸出幀數。
  int conv3d(Ctx& c,const Conv& cv,const bf16* in,int Tin,int H,int W,int C,bf16* out,int OH,int OW) {
    const int kt=int(cv.taps.size()),ts=cv.tstride,Tout=(Tin-1)/ts+1;
    MX_CHECK(kt>=1,"H3 VAE 3D 卷積缺少時間平面");
    const int K=C*cv.kh*cv.kw,N=OH*OW,O=int(cv.l.out),rows=std::min(N,std::max(4096,int((size_t(128)<<20)/(size_t(K)*2))));
    ArenaScope scope(c.act);
    auto col=c.act.alloc_n<bf16>(size_t(rows)*K);
    auto acc=c.act.alloc_n<float>(size_t(rows)*O);
    for(int t=0;t<Tout;++t) for(int start=0;start<N;start+=rows) {
      int n=std::min(rows,N-start);bool first=true;
      for(int k=0;k<kt;++k) {
        int ti=t*ts+k-(kt-1);
        if(ti<0) continue;
        h3v::im2col(in+size_t(ti)*H*W*C,col,H,W,C,cv.kh,cv.kw,cv.stride,cv.pad,OH,OW,start,n,c.stream);
        gemm_bf16(c,col,K,static_cast<const bf16*>(cv.taps[k].w.p),K,acc,true,O,n,O,K,first?0.f:1.f);
        first=false;
      }
      add_bias_f32(acc,cv.l.bias,n,O,c.stream);
      f32_to_bf16(acc,out+(size_t(t)*N+start)*O,size_t(n)*O,c.stream);
    }
    return Tout;
  }
  void resnet3d(Ctx& c,const Resnet& r,const bf16* input,bf16* output,int T,int H,int W,int C) {
    ArenaScope scope(c.act);const int N=H*W,OC=int(r.c1.l.out);
    auto norm1=c.act.alloc_n<bf16>(size_t(T)*N*C),hidden=c.act.alloc_n<bf16>(size_t(T)*N*OC);
    auto norm2=c.act.alloc_n<bf16>(size_t(T)*N*OC);
    for(int t=0;t<T;++t) h3v::groupnorm_silu(input+size_t(t)*N*C,norm1+size_t(t)*N*C,r.w1,r.b1,N,C,groups,enc_eps,c.stream);
    conv3d(c,r.c1,norm1,T,H,W,C,hidden,H,W);
    for(int t=0;t<T;++t) h3v::groupnorm_silu(hidden+size_t(t)*N*OC,norm2+size_t(t)*N*OC,r.w2,r.b2,N,OC,groups,enc_eps,c.stream);
    conv3d(c,r.c2,norm2,T,H,W,OC,output,H,W);
    if(r.shortcut.l) {conv3d(c,r.shortcut,input,T,H,W,C,hidden,H,W);add_inplace_bf16(output,hidden,size_t(T)*N*OC,c.stream);}
    else add_inplace_bf16(output,input,size_t(T)*N*OC,c.stream);
  }
  // 一段 17 幀（裝置 uint8 [T, H, W, 3]）的一個空間 tile → moments [Tl, h/16, w/16, 48]（主機）。
  std::vector<float> encode_tile3d(Ctx& c,const uint8_t* clip,int T,int H,int W,int y,int x,int h,int w,int& Tl) {
    ArenaScope scope(c.act);int max_c=*std::max_element(channels.begin(),channels.end());
    size_t capacity=size_t(T)*h*w*channels[0];int bh=h,bw=w,bt=T;
    for(size_t i=0;i<channels.size();++i) {
      capacity=std::max(capacity,size_t(bt)*bh*bw*channels[i]);
      if(levels[i].down.l) {bh=space_down[i]==2?bh/2:bh-2;bw=space_down[i]==2?bw/2:bw-2;bt=(bt-1)/time_down[i]+1;}
    }
    capacity=std::max(capacity,size_t(bt)*(h/p)*(w/p)*max_c);
    auto a=c.act.alloc_n<bf16>(capacity),b=c.act.alloc_n<bf16>(capacity);
    {
      ArenaScope s2(c.act);auto input=c.act.alloc_n<bf16>(size_t(T)*h*w*3);
      for(int t=0;t<T;++t) h3v::image_input(clip+size_t(t)*H*W*3,input+size_t(t)*h*w*3,H,W,y,x,h,w,c.stream);
      conv3d(c,conv_in,input,T,h,w,3,a,h,w);
    }
    int C=channels[0],hh=h,ww=w,tt=T;
    for(size_t i=0;i<levels.size();++i) {
      for(const auto& r:levels[i].blocks) {resnet3d(c,r,a,b,tt,hh,ww,C);std::swap(a,b);C=int(r.c1.l.out);}
      if(levels[i].down.l) {
        int oh=space_down[i]==2?hh/2:hh-2,ow=space_down[i]==2?ww/2:ww-2;
        MX_CHECK(oh>0 && ow>0,"H3 VAE encoder 空間大小不足");
        tt=conv3d(c,levels[i].down,a,tt,hh,ww,C,b,oh,ow);std::swap(a,b);hh=oh;ww=ow;
      }
    }
    for(int t=0;t<tt;++t)
      h3v::groupnorm_silu(a+size_t(t)*hh*ww*C,b+size_t(t)*hh*ww*C,enc_w,enc_b,hh*ww,C,groups,enc_eps,c.stream);
    conv3d(c,conv_out,b,tt,hh,ww,C,a,hh,ww);conv3d(c,quant,a,tt,hh,ww,48,b,hh,ww);
    const size_t n=size_t(tt)*hh*ww*48;
    auto moments=c.act.alloc_n<float>(n);bf16_to_f32(b,moments,n,c.stream);
    std::vector<float> host(n);
    MX_CUDA(cudaMemcpyAsync(host.data(),moments,n*sizeof(float),cudaMemcpyDeviceToHost,c.stream));c.sync();
    Tl=tt;return host;
  }
  size_t act_bytes(int H,int W) const {
    int h=std::min(H,256/p),w=std::min(W,256/p),T=chunk+overlap,N=T*h*w,S=N+nreg+1;
    Budget b;b.alloc(size_t(S)*D*4);b.alloc(size_t(S)*rot/2*4);b.alloc(size_t(S)*rot/2*4);size_t base=b.used;
    b.alloc(size_t(N)*24*2);b.alloc(size_t(N)*24*2);b.alloc(size_t(N)*D*2);b.used=base;
    b.alloc(size_t(S)*D*2);b.alloc(size_t(S)*3*D*2);
    for(int i=0;i<5;++i)b.alloc(size_t(S)*D*2);b.used=base;
    b.alloc(size_t(S)*D*2);b.alloc(size_t(S)*2*ff*2);b.alloc(size_t(S)*ff*2);b.alloc(size_t(S)*D*2);b.used=base;
    b.alloc(size_t(N)*D*2);b.alloc(size_t(N)*proj_out.out*2);b.alloc(size_t(T)*pt*h*p*w*p*3*4);
    return (b.peak+255)&~size_t(255);
  }
};

H3VideoVae::H3VideoVae(const TensorStore& st,int device):impl_(nullptr) {impl_=new Impl(st,device);}
H3VideoVae::~H3VideoVae(){delete impl_;}
size_t H3VideoVae::decode_act_bytes(int64_t H,int64_t W) const {
  MX_CHECK(H>0 && W>0 && H<=4096 && W<=4096,"H3 VAE latent 尺寸無效");return impl_->act_bytes(int(H),int(W));
}
void H3VideoVae::decode(Ctx& c,const float* z,int64_t F,int64_t H,int64_t W,const FrameSink& sink) {
  auto& v=*impl_;MX_CHECK(c.device==v.device && z && sink,"H3 VAE decode 裝置／指標無效");
  MX_CHECK(F>=3 && F<=100000 && H>0 && W>0 && H<=4096 && W<=4096,"H3 VAE decode 尺寸無效（至少 3 個 latent 幀）");
  c.use();ArenaScope scope(c.act);size_t gap=(256-c.act.used()%256)%256;
  MX_CHECK(c.act.capacity()-c.act.used()>=decode_act_bytes(H,W)+gap,"H3 VAE decode Arena 不足");
  int tokens=int(F)+v.drop,pad=(v.chunk-tokens%v.chunk)%v.chunk,chunks=(tokens+pad)/v.chunk-1;
  int trim=0;for(int k=0;k<pad;++k) trim+=(int(F)+k)%v.chunk==0?v.clip%v.pt:v.pt;
  int target=chunks*v.clip+v.foverlap-trim,emitted=0;
  size_t frame=size_t(H)*v.p*W*v.p*3;std::vector<float> overlap;
  auto emit=[&](const float* pixels,int n) {
    n=std::min(n,target-emitted);if(n<=0)return;
    std::vector<uint8_t> rgb(frame*n);const float mean[]={0.485f,0.456f,0.406f},std[]={0.229f,0.224f,0.225f};
    for(size_t i=0;i<rgb.size();++i) {
      float value=pixels[i]*std[i%3]+mean[i%3];MX_CHECK(std::isfinite(value),"H3 VAE decode 產生非有限值");
      rgb[i]=uint8_t(std::nearbyint(std::clamp(value,0.f,1.f)*255.f));
    }
    sink(rgb.data(),n);emitted+=n;
  };
  for(int i=0;i<chunks;++i) {
    int start=i*v.chunk,T=std::min(v.chunk+v.overlap,int(F)+pad-start);
    auto pixels=v.decode_clip(c,z,int(F),int(H),int(W),start,T);
    float* first=pixels.data()+frame*v.pre;
    if(!overlap.empty()) for(int t=0;t<v.foverlap;++t) {
      float f=float(t)/v.foverlap;
      for(size_t j=0;j<frame;++j) first[size_t(t)*frame+j]=overlap[size_t(t)*frame+j]*(1.f-f)+first[size_t(t)*frame+j]*f;
    }
    emit(first,v.clip);
    size_t tail=size_t(v.chunk*v.pt+v.pre)*frame;
    overlap.assign(pixels.begin()+tail,pixels.end());
  }
  emit(overlap.data(),int(overlap.size()/frame));MX_CHECK(emitted==target,"H3 VAE decode 影格數不符");
}
void H3VideoVae::encode_image(Ctx& c,const uint8_t* rgb,int64_t H,int64_t W,float* z) {
  auto& v=*impl_;MX_CHECK(c.device==v.device && rgb && z,"H3 VAE encode 裝置／指標無效");
  MX_CHECK(H>=32 && W>=32 && H%32==0 && W%32==0 && H<=65536 && W<=65536,"H3 VAE encode 圖片尺寸無效");
  c.use();ArenaScope scope(c.act);
  auto input=c.act.alloc_n<uint8_t>(size_t(H)*W*3);
  MX_CUDA(cudaMemcpyAsync(input,rgb,size_t(H)*W*3,cudaMemcpyHostToDevice,c.stream));
  auto ys=split_tiles(int(H),256,64,v.p),xs=split_tiles(int(W),256,64,v.p);
  int LH=int(H)/v.p,LW=int(W)/v.p;std::vector<float> moments(size_t(LH)*LW*48);
  std::vector<std::vector<float>> previous;
  for(size_t i=0;i<ys.starts.size();++i) {
    std::vector<std::vector<float>> row;
    for(size_t j=0;j<xs.starts.size();++j) {
      row.push_back(v.encode_tile(c,input,int(H),int(W),ys.starts[i],xs.starts[j],ys.lengths[i],xs.lengths[j]));
      int h=ys.lengths[i]/v.p,w=xs.lengths[j]/v.p,oh=i?ys.overlaps[i-1]/v.p:0,ow=j?xs.overlaps[j-1]/v.p:0;
      int kh=h-(i+1<ys.starts.size()?ys.overlaps[i]/v.p:0),kw=w-(j+1<xs.starts.size()?xs.overlaps[j]/v.p:0);
      stitch(row.back(),i?&previous[j]:nullptr,j?&row[j-1]:nullptr,moments,1,48,LH,LW,h,w,
          ys.starts[i]/v.p,xs.starts[j]/v.p,oh,ow,kh,kw);
    }
    previous=std::move(row);
  }
  // 使用 posterior mean；不做 pipeline 的 seed=42 抽樣與 fp16 往返。
  auto gpu=c.act.alloc_n<float>(moments.size());
  MX_CUDA(cudaMemcpyAsync(gpu,moments.data(),moments.size()*sizeof(float),cudaMemcpyHostToDevice,c.stream));
  h3v::channel_normalize(gpu,z,v.mean_buf.as<float>(),v.std_buf.as<float>(),LH*LW,c.stream);
  c.sync();
}
}  // namespace mx
namespace mx {
void H3VideoVae::encode_video(Ctx& c,const uint8_t* rgb,int64_t F,int64_t H,int64_t W,float* z) {
  auto& v=*impl_;MX_CHECK(c.device==v.device && rgb && z,"H3 VAE encode 裝置／指標無效");
  MX_CHECK(F>=1 && H>=32 && W>=32 && H%32==0 && W%32==0 && H<=4096 && W<=4096,"H3 VAE encode 影片尺寸無效");
  if(F==1) return encode_image(c,rgb,H,W,z);
  c.use();v.ensure_all_taps();ArenaScope scope(c.act);
  const size_t frame=size_t(H)*W*3;const int clips=int((F+v.clip-1)/v.clip),LH=int(H)/v.p,LW=int(W)/v.p;
  auto dclip=c.act.alloc_n<uint8_t>(frame*v.clip);
  auto ys=split_tiles(int(H),256,64,v.p),xs=split_tiles(int(W),256,64,v.p);
  std::vector<float> all;
  for(int k=0;k<clips;++k) {
    std::vector<uint8_t> host(frame*v.clip);
    for(int t=0;t<v.clip;++t) {
      int64_t src=std::min<int64_t>(int64_t(k)*v.clip+t,F-1);  // 最後一段不足 17 幀時重複最後一幀
      std::copy_n(rgb+size_t(src)*frame,frame,host.data()+size_t(t)*frame);
    }
    MX_CUDA(cudaMemcpyAsync(dclip,host.data(),host.size(),cudaMemcpyHostToDevice,c.stream));
    int Tl=0;std::vector<float> moments;std::vector<std::vector<float>> previous;
    for(size_t i=0;i<ys.starts.size();++i) {
      std::vector<std::vector<float>> row;
      for(size_t j=0;j<xs.starts.size();++j) {
        row.push_back(v.encode_tile3d(c,dclip,v.clip,int(H),int(W),ys.starts[i],xs.starts[j],ys.lengths[i],xs.lengths[j],Tl));
        if(moments.empty()) moments.resize(size_t(Tl)*LH*LW*48);
        int h=ys.lengths[i]/v.p,w=xs.lengths[j]/v.p,oh=i?ys.overlaps[i-1]/v.p:0,ow=j?xs.overlaps[j-1]/v.p:0;
        int kh=h-(i+1<ys.starts.size()?ys.overlaps[i]/v.p:0),kw=w-(j+1<xs.starts.size()?xs.overlaps[j]/v.p:0);
        stitch(row.back(),i?&previous[j]:nullptr,j?&row[j-1]:nullptr,moments,Tl,48,LH,LW,h,w,
            ys.starts[i]/v.p,xs.starts[j]/v.p,oh,ow,kh,kw);
      }
      previous=std::move(row);
    }
    all.insert(all.end(),moments.begin(),moments.end());
  }
  const int64_t Fl=encoded_frames(F),rows=Fl*LH*LW;
  MX_CHECK(int64_t(all.size())>=rows*48,"H3 VAE 影片編碼幀數不符");
  auto gpu=c.act.alloc_n<float>(size_t(rows)*48);
  MX_CUDA(cudaMemcpyAsync(gpu,all.data(),size_t(rows)*48*sizeof(float),cudaMemcpyHostToDevice,c.stream));
  h3v::channel_normalize(gpu,z,v.mean_buf.as<float>(),v.std_buf.as<float>(),int(rows),c.stream);
  c.sync();
}
}  // namespace mx
