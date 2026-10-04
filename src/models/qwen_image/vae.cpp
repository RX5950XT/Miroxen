#include "models/qwen_image/vae.h"
#include "models/qwen_image/vae_kernels.h"
#include "core/dequant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>

namespace mx {
namespace {
constexpr int kRows = 1024, kTile = 64, kHalo = 12;
constexpr size_t kWorkspace = 32 * 1024 * 1024;
struct Image { bf16* p; int h, w, c; int pixels() const { return h*w; } };
struct Conv { DevBuf weight, bias; int ci, co, k; };
struct Pool {
  bf16* a[5];
  Pool(Ctx& c, size_t n) { for (auto& p : a) p = c.act.alloc_n<bf16>(n); }
};
size_t pixels(int64_t h, int64_t w) {
  MX_CHECK(h > 0 && w > 0 && h <= INT32_MAX/16 && w <= INT32_MAX/16,
           "VAE 尺寸必須為正且不可過大");
  MX_CHECK(h <= INT32_MAX/1024/w, "VAE 圖片像素過多");
  return size_t(h)*size_t(w);
}
// 32x64 latent 的全圖 Arena 約 1.45 GiB，較大的圖片才分塊。
bool tiled(int64_t h, int64_t w) { return size_t(h)*size_t(w) > 32*64; }
size_t slot_size(int64_t h, int64_t w) {
  return pixels(h,w) * (tiled(h,w) ? 4*1152 : 256*288);
}
size_t tile_slot() { return size_t(kTile+2*kHalo)*(kTile+2*kHalo)*16*288; }
std::vector<float> read_values(const WeightRef& r) {
  MX_CHECK(dequant_supported(r.type) && !gtype_is_quant(r.type), "VAE 不支援權重型別：%s", r.name.c_str());
  MX_CHECK(r.nbytes == size_t(r.numel())*gtype_block_bytes(r.type), "VAE 權重大小錯誤：%s", r.name.c_str());
  std::vector<float> v(size_t(r.numel()));
  dequant_cpu(r.data,r.type,r.numel(),v.data());
  for(auto& x:v) {
    MX_CHECK(std::isfinite(x), "VAE 權重包含非有限值：%s",r.name.c_str());
    x=__bfloat162float(__float2bfloat16(x));  // 兩種版面皆以 bf16 執行。
  }
  return v;
}
DevBuf vector_weight(const TensorStore& st,const std::string& name,int n,int dev) {
  const auto& r=st.get(name);
  MX_CHECK(r.numel()==n && r.shape[0]==n,"VAE 向量 shape 錯誤：%s",name.c_str());
  for(size_t i=1;i<r.shape.size();++i) MX_CHECK(r.shape[i]==1,"VAE 向量 shape 錯誤：%s",name.c_str());
  auto v=read_values(r); DevBuf b(v.size()*4,dev);
  MX_CUDA(cudaMemcpy(b.get(),v.data(),b.size(),cudaMemcpyHostToDevice)); return b;
}
Conv load_conv(const TensorStore& st,const std::string& name,int ci,int co,int k,int dev) {
  const auto& r=st.get(name+".weight");
  std::vector<int64_t> s=r.shape;
  if(s.size()==5) { MX_CHECK(s[2]==1,"VAE 單幀卷積 kt 必須為 1：%s",name.c_str()); s.erase(s.begin()+2); }
  MX_CHECK((s==std::vector<int64_t>{co,ci,k,k}),"VAE 卷積 shape 錯誤：%s",name.c_str());
  auto v=read_values(r); std::vector<bf16> packed(v.size());
  for(int o=0;o<co;++o) for(int t=0;t<k*k;++t) for(int i=0;i<ci;++i)
    packed[(size_t(o)*k*k+t)*ci+i]=__float2bfloat16(v[(size_t(o)*ci+i)*k*k+t]);
  Conv cv{DevBuf(packed.size()*2,dev),vector_weight(st,name+".bias",co,dev),ci,co,k};
  MX_CUDA(cudaMemcpy(cv.weight.get(),packed.data(),cv.weight.size(),cudaMemcpyHostToDevice)); return cv;
}
}  // namespace

struct QwenImageVae::Impl {
  int device;
  bool comfy;
  std::map<std::string,Conv> convs;
  std::map<std::string,DevBuf> norms;
  Impl(const TensorStore& st,int dev):device(dev),comfy(st.has("decoder.conv1.weight")) {
    MX_CUDA(cudaSetDevice(dev));
    MX_CHECK(comfy || st.has("decoder.conv_in.weight"),"無法辨識 Qwen-Image-2.1 VAE 權重版面");
    MX_INFO("Qwen-Image-2.1 VAE 權重版面：%s",comfy?"Comfy":"diffusers");
    load(st,"post_quant_conv", "conv2",64,64,1);
    load(st,"quant_conv", "conv1",128,128,1);
    load_side(st,"decoder",64,1152,144,4,true);
    load_side(st,"encoder",4,96,768,128,false);
  }
  void load(const TensorStore& st,const std::string& key,const std::string& alt,int ci,int co,int k) {
    convs.emplace(key,load_conv(st,comfy?alt:key,ci,co,k,device));
  }
  void load_norm(const TensorStore& st,const std::string& key,const std::string& alt,int c) {
    norms.emplace(key,vector_weight(st,(comfy?alt:key)+".gamma",c,device));
  }
  void load_res(const TensorStore& st,const std::string& key,const std::string& alt,int ci,int co) {
    load_norm(st,key+".norm1",alt+".residual.0",ci);
    load(st,key+".conv1",alt+".residual.2",ci,co,3);
    load_norm(st,key+".norm2",alt+".residual.3",co);
    load(st,key+".conv2",alt+".residual.6",co,co,3);
    if(ci!=co) load(st,key+".conv_shortcut",alt+".shortcut",ci,co,1);
  }
  void load_side(const TensorStore& st,const std::string& side,int ci,int first,int last,int out,bool dec) {
    load(st,side+".conv_in",side+".conv1",ci,first,3);
    load_norm(st,side+".norm_out",side+".head.0",last);
    load(st,side+".conv_out",side+".head.2",last,out,3);
    int mid=dec?first:last;
    for(int j=0;j<2;++j) load_res(st,side+".mid_block.resnets."+std::to_string(j),
                                  side+".middle."+std::to_string(j*2),mid,mid);
    std::string a=side+".mid_block.attentions.0", b=side+".middle.1";
    load_norm(st,a+".norm",b+".norm",mid);
    load(st,a+".to_qkv",b+".to_qkv",mid,mid*3,1);
    load(st,a+".proj",b+".proj",mid,mid,1);
    int dims_d[]={1152,1152,576,288,144},dims_e[]={96,192,384,768,768};
    int current=first;
    for(int i=0;i<5;++i) {
      int co=dec?dims_d[i]:dims_e[i], blocks=dec?3:2;
      std::string key=side+(dec?".up_blocks.":".down_blocks.")+std::to_string(i);
      std::string alt=side+(dec?".upsamples.":".downsamples.")+std::to_string(i)+(dec?".upsamples.":".downsamples.");
      for(int j=0;j<blocks;++j) {
        load_res(st,key+".resnets."+std::to_string(j),alt+std::to_string(j),current,co);
        current=co;
      }
      if(i<4) load(st,key+(dec?".upsampler.resample.1":".downsampler.resample.1"),
                       alt+std::to_string(blocks)+".resample.1",co,co,3);
    }
  }
  Image conv(Ctx& c,const std::string& key,Image x,bf16* out,int stride=1,int pad=-1,bool up=false) {
    const auto& cv=convs.at(key);
    MX_CHECK(x.c==cv.ci,"VAE 卷積輸入通道錯誤：%s",key.c_str());
    if(pad<0) pad=cv.k/2;
    int oh=up?x.h*2:(stride==2?x.h/2:x.h), ow=up?x.w*2:(stride==2?x.w/2:x.w);
    ArenaScope scope(c.act);
    int rows=std::min(kRows,oh*ow), K=cv.ci*cv.k*cv.k;
    auto* cols=c.act.alloc_n<bf16>(size_t(rows)*K);
    auto* accum=c.act.alloc_n<float>(size_t(rows)*cv.co);
    for(int start=0;start<oh*ow;start+=rows) {
      int r=std::min(rows,oh*ow-start);
      qvae::columns(x.p,cols,x.h,x.w,x.c,oh,ow,cv.k,stride,pad,up,start,r,c.stream);
      gemm_bf16(c,cols,K,cv.weight.as<bf16>(),K,accum,true,cv.co,r,cv.co,K);
      qvae::finish(accum,cv.bias.as<float>(),out+size_t(start)*cv.co,r,cv.co,c.stream);
    }
    return {out,oh,ow,cv.co};
  }
  Image normalize(Ctx& c,const std::string& key,Image x,bf16* out,bool si=true) {
    qvae::norm(x.p,norms.at(key).as<float>(),out,x.pixels(),x.c,si,c.stream);
    return {out,x.h,x.w,x.c};
  }
  Image res(Ctx& c,const std::string& key,Image x,Pool& p) {
    auto n=normalize(c,key+".norm1",x,p.a[2]);
    auto y=conv(c,key+".conv1",n,p.a[3]);
    n=normalize(c,key+".norm2",y,p.a[2]);
    y=conv(c,key+".conv2",n,p.a[1]);
    auto skip=x;
    if(x.c!=y.c) skip=conv(c,key+".conv_shortcut",x,p.a[3]);
    qvae::add(y.p,skip.p,size_t(y.pixels())*y.c,c.stream);
    std::swap(p.a[0],p.a[1]); return y;
  }
  Image attention(Ctx& c,const std::string& key,Image x,Pool& p) {
    auto n=normalize(c,key+".norm",x,p.a[2],false);
    auto qkv=conv(c,key+".to_qkv",n,p.a[3]);
    int N=x.pixels(), C=x.c, chunk=std::min(128,N);
    {
      ArenaScope scope(c.act);
      auto* scores=c.act.alloc_n<float>(size_t(chunk)*N);
      auto* value=c.act.alloc_n<float>(size_t(N)*C);
      auto* out=c.act.alloc_n<float>(size_t(chunk)*C);
      qvae::attention_value(qkv.p,value,N,C,c.stream);
      for(int start=0;start<N;start+=chunk) {
        int r=std::min(chunk,N-start);
        gemm_bf16(c,qkv.p+size_t(start)*3*C,3*C,qkv.p+C,3*C,scores,true,N,r,N,C,0,1.f/std::sqrt(float(C)));
        qvae::softmax(scores,r,N,c.stream);
        float alpha=1,beta=0;
        MX_BLAS(cublasSgemm(c.blas,CUBLAS_OP_N,CUBLAS_OP_N,C,r,N,&alpha,value,C,
                            scores,N,&beta,out,C));
        qvae::attention_out(out,p.a[2]+size_t(start)*C,size_t(r)*C,c.stream);
      }
    }
    auto y=conv(c,key+".proj",{p.a[2],x.h,x.w,C},p.a[1]);
    qvae::add(y.p,x.p,size_t(N)*C,c.stream);
    std::swap(p.a[0],p.a[1]); return y;
  }
  Image mid(Ctx& c,const std::string& side,Image x,Pool& p) {
    x=res(c,side+".mid_block.resnets.0",x,p);
    x=attention(c,side+".mid_block.attentions.0",x,p);
    return res(c,side+".mid_block.resnets.1",x,p);
  }
  Image block(Ctx& c,Image x,Pool& p,int i,bool dec,bf16* resample_out=nullptr) {
    std::string key=(dec?"decoder.up_blocks.":"encoder.down_blocks.")+std::to_string(i);
    Image original{p.a[4],x.h,x.w,x.c};
    if(!dec || i<4)
      MX_CUDA(cudaMemcpyAsync(original.p,x.p,size_t(x.pixels())*x.c*2,cudaMemcpyDeviceToDevice,c.stream));
    for(int j=0;j<(dec?3:2);++j) x=res(c,key+".resnets."+std::to_string(j),x,p);
    if(i<4) {
      x=conv(c,key+(dec?".upsampler.resample.1":".downsampler.resample.1"),x,
             resample_out?resample_out:p.a[1],dec?1:2,dec?1:0,dec);
      if(!resample_out) std::swap(p.a[0],p.a[1]);
    }
    if(!dec || i<4) qvae::shortcut(x.p,original.p,original.h,original.w,original.c,x.c,
                                  i<4?2:1,dec?(i<3?2:1):((i>0&&i<4)?2:1),dec,c.stream);
    return x;
  }
  Image tail(Ctx& c,Image x,Pool& p) {
    for(int i=2;i<5;++i) x=block(c,x,p,i,true);
    auto n=normalize(c,"decoder.norm_out",x,p.a[2]);
    return conv(c,"decoder.conv_out",n,p.a[1]);
  }
};

QwenImageVae::QwenImageVae(const TensorStore& st,int device):impl_(nullptr) {
  auto impl=std::make_unique<Impl>(st,device); impl_=impl.release();
}
QwenImageVae::~QwenImageVae() { delete impl_; }
size_t QwenImageVae::decode_act_bytes(int64_t H,int64_t W) const {
  size_t n=slot_size(H,W), tiles=tiled(H,W)?tile_slot()*5*2+pixels(H,W)*16*1152*2:0;
  size_t attn=pixels(H,W)*(1152+128)*4+128*1152*4;
  return n*5*2+tiles+std::max(kWorkspace,attn)+pixels(H,W)*256*4+8192;
}
void QwenImageVae::decode(Ctx& c,const float* z,int64_t H,int64_t W,std::vector<uint8_t>& rgba) {
  size_t count=pixels(H,W);
  MX_CHECK(z && c.device==impl_->device,"VAE decode 的指標或 GPU 不正確");
  MX_CHECK(c.act.capacity()-c.act.used()>=decode_act_bytes(H,W),"VAE decode Arena 空間不足");
  c.use(); ArenaScope scope(c.act); Pool p(c,slot_size(H,W));
  Image x{p.a[0],int(H),int(W),64}; qvae::latent_in(z,x.p,int(count),c.stream);
  x=impl_->conv(c,"post_quant_conv",x,p.a[1]); std::swap(p.a[0],p.a[1]);
  x=impl_->conv(c,"decoder.conv_in",x,p.a[1]); std::swap(p.a[0],p.a[1]);
  x=impl_->mid(c,"decoder",x,p);
  x=impl_->block(c,x,p,0,true);
  // 第二次放大是前段最大的一張圖；後續分塊只讀它，無須將五個暫存都放大。
  bf16* base=tiled(H,W)?c.act.alloc_n<bf16>(count*16*1152):nullptr;
  x=impl_->block(c,x,p,1,true,base);
  auto* output=c.act.alloc_n<uint8_t>(count*256*4);
  if(!tiled(H,W)) {
    auto y=impl_->tail(c,x,p);
    qvae::image_out(y.p,output,y.h,y.w,int(W)*16,0,0,y.h,y.w,0,0,c.stream);
  } else {
    // 最後三個區塊所需的 halo 為左側 12、右側 11（以此層像素計）。
    // 丟棄重疊區，數學上等同全圖，無須混合；保留全域 attention。
    // 32x40 與全圖實測：RGBA RMSE 0.11659、PSNR 66.80 dB、max 35；
    // 差異來自 cuBLAS 分塊尺寸改變後的 bf16 取整，並非截斷依賴範圍。
    Pool t(c,tile_slot());
    for(int top=0;top<x.h;top+=kTile) for(int left=0;left<x.w;left+=kTile) {
      int y0=std::max(0,top-kHalo),x0=std::max(0,left-kHalo);
      int y1=std::min(x.h,top+kTile+kHalo),x1=std::min(x.w,left+kTile+kHalo);
      Image tile{t.a[0],y1-y0,x1-x0,x.c};
      qvae::crop(x.p,tile.p,x.w,x.c,y0,x0,tile.h,tile.w,c.stream);
      auto y=impl_->tail(c,tile,t);
      qvae::image_out(y.p,output,y.h,y.w,int(W)*16,(top-y0)*4,(left-x0)*4,
           std::min(kTile,x.h-top)*4,std::min(kTile,x.w-left)*4,top*4,left*4,c.stream);
    }
  }
  rgba.resize(count*256*4);
  MX_CUDA(cudaMemcpyAsync(rgba.data(),output,rgba.size(),cudaMemcpyDeviceToHost,c.stream)); c.sync();
}
void QwenImageVae::encode(Ctx& c,const uint8_t* rgba,int64_t h,int64_t w,float* z) {
  MX_CHECK(rgba && z && h>0 && w>0 && h%16==0 && w%16==0,"VAE encode 需要 RGBA 指標及 16 倍數尺寸");
  pixels(h/16,w/16);
  MX_CHECK(c.device==impl_->device,"VAE encode GPU 不正確");
  size_t count=size_t(h)*w,n=count*96;
  MX_CHECK(c.act.capacity()-c.act.used()>=n*5*2+count*4+kWorkspace+8192,"VAE encode Arena 空間不足");
  c.use(); ArenaScope scope(c.act); Pool p(c,n);
  auto* input=c.act.alloc_n<uint8_t>(count*4);
  MX_CUDA(cudaMemcpyAsync(input,rgba,count*4,cudaMemcpyHostToDevice,c.stream));
  qvae::image_in(input,p.a[0],int(count),c.stream);
  Image x=impl_->conv(c,"encoder.conv_in",{p.a[0],int(h),int(w),4},p.a[1]); std::swap(p.a[0],p.a[1]);
  for(int i=0;i<5;++i) x=impl_->block(c,x,p,i,false);
  x=impl_->mid(c,"encoder",x,p);
  auto norm=impl_->normalize(c,"encoder.norm_out",x,p.a[2]);
  x=impl_->conv(c,"encoder.conv_out",norm,p.a[1]); std::swap(p.a[0],p.a[1]);
  x=impl_->conv(c,"quant_conv",x,p.a[1]);
  qvae::latent_out(x.p,z,x.pixels(),c.stream); c.sync();
}

const float* QwenImageVae::latents_mean() {
  static const float v[]={0.5126f,0.7721f,-0.0631f,1.3506f,-0.7855f,-2.1025f,-0.3458f,1.3722f,1.8873f,-1.7177f,-0.651f,0.2732f,0.7562f,-0.6163f,-1.0277f,3.8363f,2.021f,0.0472f,0.932f,2.0087f,2.4954f,-0.1391f,-1.4249f,1.8464f,-0.5236f,1.2826f,3.7046f,-1.3035f,2.7286f,-1.4518f,-1.9036f,-1.9955f,-0.0342f,-1.0265f,-0.7636f,3.0555f,0.0746f,-3.0751f,-0.1076f,1.7376f,-1.0914f,-1.9435f,-0.2784f,-1.368f,0.4809f,-0.4433f,0.3764f,0.5729f,-2.0595f,1.096f,-1.326f,-2.0211f,-5.0179f,0.5275f,4.0162f,1.8505f,0.3026f,1.9373f,1.4937f,0.2632f,0.5547f,-1.7121f,-0.1562f,0.0304f};
  return v;
}
const float* QwenImageVae::latents_std() {
  static const float v[]={3.2001f,3.2936f,3.4321f,3.0091f,3.1061f,4.0379f,4.0705f,3.791f,3.0785f,3.65f,3.9308f,3.0904f,2.8778f,3.7675f,3.732f,5.0756f,3.2864f,4.0397f,3.1317f,4.0443f,2.9249f,3.9454f,3.0988f,4.2489f,3.4896f,3.8513f,3.9323f,3.4719f,3.7498f,4.283f,3.5694f,4.2467f,3.9037f,3.2947f,5.077f,3.5075f,3.27f,3.4767f,2.8063f,5.1125f,3.5327f,4.7833f,3.1286f,4.1819f,3.8527f,3.8312f,3.5605f,4.3875f,3.9624f,4.0168f,3.5643f,4.055f,5.5614f,4.2963f,4.408f,3.4959f,3.8747f,3.7608f,3.5735f,3.149f,3.7662f,3.6746f,3.4563f,3.8161f};
  return v;
}
}  // namespace mx
