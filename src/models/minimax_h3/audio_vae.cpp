#include "models/minimax_h3/audio_vae.h"
#include "models/minimax_h3/audio_vae_kernels.h"
#include "core/dequant.h"
#include "core/weights.h"
#include "kernels/ops.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>

namespace mx {
namespace {
// 只讀此模型 config 的數字欄位，不引入額外 JSON 相依套件。
struct Number {
  double value = 0;
  bool array = false;
  std::vector<Number> items;
};
void whitespace(const char*& p) { while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p; }
Number number(const char*& p, int depth = 0) {
  MX_CHECK(depth < 4, "音訊 config 陣列過深");
  whitespace(p); Number n;
  if (*p == '[') {
    n.array = true; ++p; whitespace(p);
    if (*p == ']') { ++p; return n; }
    while (true) {
      n.items.push_back(number(p, depth + 1)); whitespace(p);
      if (*p == ']') { ++p; break; }
      MX_CHECK(*p++ == ',', "音訊 config 陣列格式錯誤");
    }
  } else {
    char* end = nullptr;
    MX_CHECK(*p == '-' || (*p >= '0' && *p <= '9'), "音訊 config 需要數字");
    n.value = std::strtod(p, &end);
    MX_CHECK(end != p && std::isfinite(n.value), "音訊 config 數字無效"); p = end;
  }
  return n;
}
Number field(const std::string& text, const std::string& key) {
  std::regex pattern("\"" + key + "\"\\s*:");
  auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
  MX_CHECK(begin != std::sregex_iterator(), "音訊 config 缺少 %s", key.c_str());
  auto hit = *begin;
  MX_CHECK(++begin == std::sregex_iterator(), "音訊 config 重複 %s", key.c_str());
  const char* p = text.c_str() + hit.position() + hit.length();
  Number n = number(p); whitespace(p);
  MX_CHECK(*p == ',' || *p == '}', "音訊 config 欄位 %s 格式錯誤", key.c_str());
  return n;
}
int positive(const Number& n) {
  MX_CHECK(!n.array && n.value >= 1 && n.value <= 65536 && std::floor(n.value) == n.value,
           "音訊 config 需要 1..65536 的整數");
  return int(n.value);
}
std::vector<int> integers(const Number& n) {
  MX_CHECK(n.array && !n.items.empty() && n.items.size() <= 32, "音訊 config 整數陣列無效");
  std::vector<int> v; for (const auto& x : n.items) v.push_back(positive(x)); return v;
}
std::vector<float> floats(const Number& n, bool stddev) {
  MX_CHECK(n.array && n.items.size() == 32, "音訊 config latent 統計必須有 32 個值");
  std::vector<float> v;
  for (const auto& x : n.items) {
    MX_CHECK(!x.array && std::isfinite(float(x.value)) && (!stddev || float(x.value) > 0), "音訊 config latent 統計無效");
    v.push_back(float(x.value));
  }
  return v;
}
std::string config_text(const TensorStore& st) {
  // 固定的 public constructor 沒有路徑參數；呼叫端可用 metadata 或環境變數指定旁邊的 config。
  auto text = st.meta_str("h3_audio_vae_config");
  if (!text.empty()) {
    MX_CHECK(text.size() <= (1u << 20), "音訊 config metadata 大小無效"); return text;
  }
  const char* path = std::getenv("MX_H3_AUDIO_VAE_CONFIG");
  std::ifstream f(path ? path : "models/minimax-h3/hf/audio_vae/config.json", std::ios::binary);
  MX_CHECK(f.good(), "無法讀取音訊 config；請設定 MX_H3_AUDIO_VAE_CONFIG");
  std::ostringstream out; out << f.rdbuf(); text = out.str();
  MX_CHECK(!text.empty() && text.size() <= (1u << 20), "音訊 config 大小無效"); return text;
}
struct Config {
  int enc, latent, dec, heads, hop = 1;
  std::vector<int> er, dr, dk, rk;
  std::vector<std::vector<int>> rd;
  std::vector<float> mean, std;
  explicit Config(const TensorStore& st) {
    auto t = config_text(st);
    enc = positive(field(t,"encoder_dim")); latent = positive(field(t,"latent_dim"));
    dec = positive(field(t,"decoder_dim")); heads = positive(field(t,"num_attention_heads"));
    MX_CHECK(positive(field(t,"latent_channels")) == 32 && positive(field(t,"sampling_rate")) == 32000,
             "音訊 VAE 介面只支援 32 channels / 32000 Hz");
    er=integers(field(t,"encoder_rates")); dr=integers(field(t,"decoder_rates"));
    dk=integers(field(t,"decoder_kernel_sizes")); rk=integers(field(t,"resblock_kernel_sizes"));
    auto dilation=field(t,"resblock_dilation_sizes");
    MX_CHECK(dilation.array, "音訊 config resblock_dilation_sizes 需要陣列");
    for(const auto& d:dilation.items)rd.push_back(integers(d));
    MX_CHECK(dr.size()==dk.size() && rk.size()==rd.size(), "音訊 config decoder 陣列長度不同");
    int64_t down=1,up=1;
    for(int r:er){down*=r;MX_CHECK(down<=800,"音訊 encoder hop 過大");}
    for(size_t i=0;i<dr.size();i++){
      up*=dr[i]; MX_CHECK(up<=800 && dk[i]>=dr[i] && (dk[i]-dr[i])%2==0,"音訊 decoder rate/kernel 無效");
    }
    MX_CHECK(down==800 && up==down && latent%heads==0 && latent%32==0,
             "音訊 config hop 或 attention 維度無效"); hop=int(down);
    MX_CHECK(er.size()<16 && dr.size()<16 && int64_t(enc)*(1LL<<er.size())<=65536 &&
             dec%(1<<dr.size())==0, "音訊 config 通道數無效");
    for(int k:rk)MX_CHECK(k%2==1,"音訊 resblock kernel 必須為奇數");
    mean=floats(field(t,"latents_mean"),false); std=floats(field(t,"latents_std"),true);
  }
};
struct Signal {
  DevBuf buf;
  int C, T;
  Signal(int channels,int time,int dev):buf(size_t(channels)*time*sizeof(float),dev),C(channels),T(time) {}
  float* p() const { return buf.as<float>(); }
};
struct Conv {
  DevBuf storage;
  const float* w = nullptr;
  const float* b = nullptr;
  std::vector<int64_t> shape;
};
DevBuf upload(const std::vector<float>& v,int dev) {
  DevBuf b(v.size()*sizeof(float),dev);
  MX_CUDA(cudaMemcpy(b.get(),v.data(),b.size(),cudaMemcpyHostToDevice));return b;
}
std::vector<float> cpu_weight(const WeightRef& r) {
  MX_CHECK(!r.shape.empty() && r.numel()>0 && r.numel()<INT32_MAX && dequant_supported(r.type),
           "音訊權重 %s 大小或型別無效",r.name.c_str());
  std::vector<float> v(size_t(r.numel())); dequant_cpu(r.data,r.type,r.numel(),v.data());
  for(float x:v)MX_CHECK(std::isfinite(x),"音訊權重 %s 有非有限數字",r.name.c_str());return v;
}
}

struct H3AudioVae::Impl {
  Config cfg;
  int dev;
  DeviceWeights dw;
  std::map<std::string,Conv> convs;
  std::map<std::string,Linear> lines;
  DevBuf mean,std,qkv_bias;

  Impl(const TensorStore& st,int device):cfg(st),dev(device),dw(st,device) {
    MX_CUDA(cudaSetDevice(dev));
    mean=upload(cfg.mean,dev);std=upload(cfg.std,dev);
    // 在載入時摺合 weight norm；ConvTranspose1d 也是沿第 0 維正規化。
    for(const auto& name:st.names()) {
      if(name.ends_with(".weight_v") || (name.ends_with(".weight") && st.get(name).shape.size()==3)) {
        std::string prefix=name.substr(0,name.rfind('.'));
        const auto& r=st.get(name);Conv cv;cv.shape=r.shape;
        MX_CHECK(cv.shape.size()==3,"音訊卷積 %s 需要 3 維權重",prefix.c_str());
        for(auto d:cv.shape)MX_CHECK(d>0 && d<=65536,"音訊卷積 shape 無效");
        auto w=cpu_weight(r);
        if(name.ends_with(".weight_v")) {
          const auto& gref=st.get(prefix+".weight_g");
          MX_CHECK(gref.shape==std::vector<int64_t>({cv.shape[0],1,1}),"音訊 weight_g shape 無效");
          auto g=cpu_weight(gref);int64_t row=cv.shape[1]*cv.shape[2];
          for(int64_t i=0;i<cv.shape[0];i++) {
            double sum=0;for(int64_t j=0;j<row;j++)sum+=double(w[i*row+j])*w[i*row+j];
            MX_CHECK(sum>0,"音訊 weight norm 長度為零");
            float scale=g[i]/float(std::sqrt(sum));
            for(int64_t j=0;j<row;j++)w[i*row+j]*=scale;
          }
        }
        cv.storage=upload(w,dev);cv.w=cv.storage.as<float>();
        if(st.has(prefix+".bias")) {
          const auto& b=st.get(prefix+".bias");MX_CHECK(b.shape.size()==1,"音訊 bias shape 無效");
          cpu_weight(b);cv.b=dw.f32(prefix+".bias");
        }
        convs.emplace(prefix,std::move(cv));
      } else if(name.ends_with(".weight") && st.get(name).shape.size()==2) {
        cpu_weight(st.get(name));
        auto prefix=name.substr(0,name.rfind('.'));
        if(st.has(prefix+".bias")) {
          MX_CHECK(st.get(prefix+".bias").shape==std::vector<int64_t>({st.get(name).shape[0]}),
                   "音訊 linear bias %s 大小錯誤",prefix.c_str());
          cpu_weight(st.get(prefix+".bias"));
        }
        lines.emplace(prefix,make_linear(dw,prefix));
      } else if(name.ends_with(".alpha") || name.ends_with(".beta") || name.ends_with(".filter") ||
                name.starts_with("pre_block.")) {
        cpu_weight(st.get(name));dw.f32(name);
      }
    }
    auto qb=cpu_weight(st.get("pre_block.attn.q_bias"));
    auto vb=cpu_weight(st.get("pre_block.attn.v_bias"));
    MX_CHECK(qb.size()==size_t(cfg.latent) && vb.size()==qb.size(),"音訊 Q/V bias 大小無效");
    std::vector<float> b(3*cfg.latent,0.f);std::copy(qb.begin(),qb.end(),b.begin());
    if(st.has("pre_block.attn.zero_k_bias")) {
      auto kb=cpu_weight(st.get("pre_block.attn.zero_k_bias"));
      MX_CHECK(kb.size()==qb.size(),"音訊 K bias 大小無效");std::copy(kb.begin(),kb.end(),b.begin()+cfg.latent);
    }
    std::copy(vb.begin(),vb.end(),b.begin()+2*cfg.latent);qkv_bias=upload(b,dev);
  }
  const float* vector(const std::string& n,int count) {
    MX_CHECK(dw.store().get(n).numel()==count,"音訊權重 %s 大小無效",n.c_str());return dw.f32(n);
  }
  Signal conv(Ctx& c,const Signal& x,const std::string& name,int stride=1,int pad=-1,int dil=1,bool trans=false) {
    auto it=convs.find(name);MX_CHECK(it!=convs.end(),"音訊缺少卷積 %s",name.c_str());
    const auto& w=it->second;int in=int(w.shape[trans?0:1]),out=int(w.shape[trans?1:0]),k=int(w.shape[2]);
    MX_CHECK(x.C==in,"音訊卷積 %s 輸入通道錯誤",name.c_str());if(pad<0)pad=(k-1)*dil/2;
    int64_t length=trans?(int64_t(x.T)-1)*stride-2*pad+k:(int64_t(x.T)+2*pad-dil*(k-1)-1)/stride+1;
    MX_CHECK(length>0 && length<INT32_MAX && int64_t(in)*k<INT32_MAX,"音訊卷積長度無效");
    if(w.b)MX_CHECK(dw.store().get(name+".bias").numel()==out,"音訊卷積 bias 大小錯誤");
    Signal y(out,int(length),dev);float alpha=1.f,beta=0.f;
    if(trans) {
      Signal col(out*k,x.T,dev);
      MX_BLAS(cublasSgemm(c.blas,CUBLAS_OP_N,CUBLAS_OP_T,x.T,out*k,in,&alpha,x.p(),x.T,w.w,out*k,&beta,col.p(),x.T));
      h3audio::col2im(col.p(),y.p(),w.b,out,x.T,y.T,k,stride,pad,c.stream);
    } else {
      Signal col(in*k,y.T,dev);
      h3audio::im2col(x.p(),col.p(),in,x.T,y.T,k,stride,pad,dil,c.stream);
      MX_BLAS(cublasSgemm(c.blas,CUBLAS_OP_N,CUBLAS_OP_N,y.T,out,in*k,&alpha,col.p(),y.T,w.w,in*k,&beta,y.p(),y.T));
      h3audio::bias(y.p(),w.b,out,y.T,c.stream);
    }
    return y;
  }
  Signal activate(Ctx& c,const Signal& x,const std::string& n,bool alias) {
    Signal y(x.C,x.T,dev);
    if(!alias)h3audio::snake(x.p(),y.p(),vector(n+".alpha",x.C),x.C,x.T,c.stream);
    else {
      int k=int(dw.store().get(n+".upsample.filter").numel());
      MX_CHECK(k>=4 && k%2==0 && k<=128 && dw.store().get(n+".downsample.lowpass.filter").numel()==k,
               "音訊 alias-free filter 大小無效");
      Signal up(x.C,x.T*2,dev);
      h3audio::alias_up(x.p(),up.p(),vector(n+".upsample.filter",k),vector(n+".act.alpha",x.C),
                       vector(n+".act.beta",x.C),x.C,x.T,k,c.stream);
      h3audio::alias_down(up.p(),y.p(),vector(n+".downsample.lowpass.filter",k),x.C,x.T,k,c.stream);
    }
    return y;
  }
  Signal amp(Ctx& c,const Signal& input,const std::string& n,const std::vector<int>& dilation) {
    Signal x(input.C,input.T,dev);MX_CUDA(cudaMemcpyAsync(x.p(),input.p(),x.buf.size(),cudaMemcpyDeviceToDevice,c.stream));
    for(size_t j=0;j<dilation.size();j++) {
      auto a=activate(c,x,n+".activations."+std::to_string(2*j),true);
      auto h=conv(c,a,n+".convs1."+std::to_string(j),1,-1,dilation[j]);
      a=activate(c,h,n+".activations."+std::to_string(2*j+1),true);
      h=conv(c,a,n+".convs2."+std::to_string(j));
      MX_CHECK(h.C==x.C && h.T==x.T,"音訊 AMP residual shape 錯誤");
      axpy_f32(x.p(),h.p(),1.f,int64_t(x.C)*x.T,c.stream);
    }
    return x;
  }
  Signal decode(Ctx& c,const float* z,int A) {
    Signal input(32,A,dev);h3audio::normalize(z,input.p(),mean.as<float>(),std.as<float>(),A,false,c.stream);
    auto x=conv(c,input,"dec_in_proj");x=conv(c,x,"decoder.conv_pre");
    MX_CHECK(x.C==cfg.dec,"音訊 decoder_dim 與權重不同");
    for(size_t i=0;i<cfg.dr.size();i++) {
      auto name="decoder.ups."+std::to_string(i)+".0";
      MX_CHECK(convs.at(name).shape[2]==cfg.dk[i],"音訊 decoder kernel 與 config 不同");
      x=conv(c,x,name,cfg.dr[i],(cfg.dk[i]-cfg.dr[i])/2,1,true);
      Signal sum(x.C,x.T,dev);MX_CUDA(cudaMemsetAsync(sum.p(),0,sum.buf.size(),c.stream));
      for(size_t j=0;j<cfg.rk.size();j++) {
        auto n="decoder.resblocks."+std::to_string(i*cfg.rk.size()+j);
        MX_CHECK(convs.at(n+".convs1.0").shape[2]==cfg.rk[j],"音訊 resblock kernel 與 config 不同");
        auto h=amp(c,x,n,cfg.rd[j]);axpy_f32(sum.p(),h.p(),1.f/float(cfg.rk.size()),int64_t(x.C)*x.T,c.stream);
      }
      x=std::move(sum);
    }
    auto h=activate(c,x,"decoder.activation_post",true);return conv(c,h,"decoder.conv_post");
  }
  Signal ln(Ctx& c,const Signal& x,const std::string& n) {
    Signal y(x.C,x.T,dev);
    h3audio::norm(x.p(),y.p(),vector(n+".weight",x.C),vector(n+".bias",x.C),x.T,x.C,c.stream);return y;
  }
  Signal linear(Ctx& c,const Signal& x,const std::string& n) {
    auto it=lines.find(n);MX_CHECK(it!=lines.end() && it->second.in==x.C,"音訊 linear %s 維度錯誤",n.c_str());
    Signal y(int(it->second.out),x.T,dev);linear_f32(c,it->second,x.p(),x.T,y.p());return y;
  }
  Signal attention(Ctx& c,const Signal& x) {
    auto q=linear(c,x,"pre_block.attn.qkv");int T=x.T,D=cfg.latent/cfg.heads;
    MX_CHECK(q.C==3*cfg.latent,"音訊 qkv shape 錯誤");
    add_bias_f32(q.p(),qkv_bias.as<float>(),T,q.C,c.stream);
    Signal scores(T,T,dev),heads(cfg.heads*D,T,dev);float alpha=1.f,beta=0.f;
    for(int h=0;h<cfg.heads;h++) {
      const float* qp=q.p()+h*D;const float* kp=qp+cfg.latent;const float* vp=kp+cfg.latent;
      MX_BLAS(cublasSgemm(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,T,T,D,&alpha,kp,q.C,qp,q.C,&beta,scores.p(),T));
      h3audio::softmax(scores.p(),T,D,c.stream);
      MX_BLAS(cublasSgemm(c.blas,CUBLAS_OP_N,CUBLAS_OP_N,D,T,T,&alpha,vp,q.C,scores.p(),T,&beta,heads.p()+int64_t(h)*T*D,D));
    }
    Signal pooled(32,T,dev);h3audio::pool(heads.p(),pooled.p(),T,cfg.heads,D,32,c.stream);
    return linear(c,pooled,"pre_block.attn.proj");
  }
  Signal project(Ctx& c,const Signal& input) {
    Signal tokens(input.C,input.T,dev);h3audio::transpose(input.p(),tokens.p(),input.C,input.T,c.stream);
    auto n=ln(c,tokens,"pre_block.norm3");auto x=linear(c,n,"pre_block.proj");
    n=ln(c,tokens,"pre_block.norm1");auto a=attention(c,n);
    axpy_f32(x.p(),a.p(),1.f,int64_t(x.C)*x.T,c.stream);
    n=ln(c,x,"pre_block.norm2");n=ln(c,n,"pre_block.mlp.norm");
    auto gate=linear(c,n,"pre_block.mlp.w0"),up=linear(c,n,"pre_block.mlp.w1");
    MX_CHECK(gate.C==up.C,"音訊 GeGLU shape 不同");h3audio::geglu(gate.p(),up.p(),int64_t(gate.C)*gate.T,c.stream);
    a=linear(c,gate,"pre_block.mlp.w2");axpy_f32(x.p(),a.p(),1.f,int64_t(x.C)*x.T,c.stream);
    Signal channels(x.C,x.T,dev);h3audio::transpose(x.p(),channels.p(),x.T,x.C,c.stream);return channels;
  }
  Signal encode(Ctx& c,const std::vector<float>& mono) {
    Signal input(1,int(mono.size()),dev);MX_CUDA(cudaMemcpyAsync(input.p(),mono.data(),input.buf.size(),cudaMemcpyHostToDevice,c.stream));
    auto x=conv(c,input,"encoder.block.0");MX_CHECK(x.C==cfg.enc,"音訊 encoder_dim 與權重不同");
    const int dilation[]={1,3,9};
    for(size_t i=0;i<cfg.er.size();i++) {
      auto n="encoder.block."+std::to_string(i+1)+".block";
      for(int j=0;j<3;j++) {
        auto unit=n+"."+std::to_string(j)+".block";
        auto a=activate(c,x,unit+".0",false);auto h=conv(c,a,unit+".1",1,-1,dilation[j]);
        a=activate(c,h,unit+".2",false);h=conv(c,a,unit+".3");
        MX_CHECK(h.C==x.C && h.T==x.T,"音訊 encoder residual shape 錯誤");
        axpy_f32(x.p(),h.p(),1.f,int64_t(x.C)*x.T,c.stream);
      }
      auto a=activate(c,x,n+".3",false);x=conv(c,a,n+".4",cfg.er[i],(cfg.er[i]+1)/2);
    }
    auto n="encoder.block."+std::to_string(cfg.er.size()+1);
    auto a=activate(c,x,n,false);x=conv(c,a,"encoder.block."+std::to_string(cfg.er.size()+2));
    MX_CHECK(x.C==cfg.latent,"音訊 latent_dim 與權重不同");x=project(c,x);return conv(c,x,"mean_proj");
  }
};

H3AudioVae::H3AudioVae(const TensorStore& st,int device):impl_(nullptr) {
  auto p=std::make_unique<Impl>(st,device);impl_=p.release();
}
H3AudioVae::~H3AudioVae(){delete impl_;}
void H3AudioVae::decode(Ctx& c,const float* z,int64_t A,std::vector<float>& pcm) {
  MX_CHECK(z && A>0 && A<=INT32_MAX/1600 && c.device==impl_->dev,"音訊 decode 輸入或裝置無效");c.use();
  int64_t samples=A*impl_->cfg.hop;pcm.resize(size_t(samples)*2);
  for(int ch=0;ch<2;ch++) {
    auto x=impl_->decode(c,z+ch*32*A,int(A));
    MX_CHECK(x.C==1 && x.T==samples,"音訊 decoder 輸出 shape 錯誤");
    std::vector<float> mono(size_t(samples),0.f);
    MX_CUDA(cudaMemcpyAsync(mono.data(),x.p(),x.buf.size(),cudaMemcpyDeviceToHost,c.stream));c.sync();
    for(int64_t i=0;i<samples;i++) {
      MX_CHECK(std::isfinite(mono[i]),"音訊 decode 產生非有限數字");pcm[2*i+ch]=std::clamp(mono[i],-1.f,1.f);
    }
  }
}
void H3AudioVae::encode(Ctx& c,const float* pcm,int64_t samples,float* z,int64_t& A) {
  MX_CHECK(pcm && z && samples>0 && samples<=INT32_MAX/2-800 && c.device==impl_->dev,"音訊 encode 輸入或裝置無效");c.use();
  // diffusers 先右側補零；非整數 hop 的音訊採 ceil，與 header 的 round 說明不同。
  int64_t frames=(samples+impl_->cfg.hop-1)/impl_->cfg.hop;
  for(int ch=0;ch<2;ch++) {
    std::vector<float> mono(size_t(frames*impl_->cfg.hop),0.f);
    for(int64_t i=0;i<samples;i++) {
      MX_CHECK(std::isfinite(pcm[2*i+ch]),"音訊 PCM 有非有限數字");mono[i]=pcm[2*i+ch];
    }
    auto x=impl_->encode(c,mono);MX_CHECK(x.C==32 && x.T==frames,"音訊 encoder 輸出 shape 錯誤");
    h3audio::normalize(x.p(),z+ch*32*frames,impl_->mean.as<float>(),impl_->std.as<float>(),x.T,true,c.stream);c.sync();
  }
  A=frames;
}
}
