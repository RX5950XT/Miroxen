// 完整權重與 diffusers 的數值比對；缺檔時明確跳過。
#include "models/minimax_h3/audio_vae.h"
#include "core/dequant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <vector>

using namespace mx;
namespace fs=std::filesystem;
namespace {
fs::path model_root() {
  if(const char* p=std::getenv("MX_MODELS_DIR"))return p;
  for(const auto& p:{fs::path("models"),fs::path("../../Miroxen/models")})
    if(fs::exists(p/"video/minimax-h3/hf/audio_vae/config.json"))return fs::absolute(p);
  return "models";
}
std::vector<float> read(const TensorStore& st,const std::string& n,const std::vector<int64_t>& shape) {
  const auto& r=st.get(n);MX_CHECK(r.shape==shape,"參考 %s shape 錯誤",n.c_str());
  std::vector<float> v(size_t(r.numel()));dequant_cpu(r.data,r.type,r.numel(),v.data());
  for(float x:v)MX_CHECK(std::isfinite(x),"參考有非有限數字");return v;
}
double relative(const std::vector<float>& got,const std::vector<float>& expected) {
  MX_CHECK(got.size()==expected.size(),"測試長度不同");double n=0,d=0;
  for(size_t i=0;i<got.size();i++) {
    if(!std::isfinite(got[i]))return INFINITY;
    double e=double(got[i])-expected[i];n+=e*e;d+=double(expected[i])*expected[i];
  }
  return std::sqrt(n/std::max(d,1e-30));
}
void putenv(const char* key,const std::string& value) {
#ifdef _WIN32
  MX_CHECK(_putenv_s(key,value.c_str())==0,"無法設定測試環境變數");
#else
  MX_CHECK(setenv(key,value.c_str(),1)==0,"無法設定測試環境變數");
#endif
}
template<class F> bool rejects(F f) {
  try{f();return false;}catch(const std::runtime_error&){return true;}
}
}
int main() {
  try {
    putenv("CUDA_DEVICE_ORDER","PCI_BUS_ID");auto root=model_root();
    auto ref=root/"ref/h3_audio_vae_ref.safetensors",cfg=root/"video/minimax-h3/hf/audio_vae/config.json";
    if(!fs::exists(ref) || !fs::exists(cfg)){std::printf("SKIP：缺少參考或 config：%s\n",root.string().c_str());return 0;}
    if(!std::getenv("MX_H3_AUDIO_VAE_CONFIG"))putenv("MX_H3_AUDIO_VAE_CONFIG",cfg.string());
    TensorStore rs;rs.add_file(ref.string());
    auto latent=read(rs,"normalized",{2,32,80}),pcm=read(rs,"pcm",{64000,2});
    auto expected=read(rs,"decoded",{2,1,64000});
    std::vector<float> stereo(expected.size());
    for(int i=0;i<64000;i++){stereo[2*i]=expected[i];stereo[2*i+1]=expected[64000+i];}
    auto encoded=read(rs,"encoded",{2,32,80});
    auto short_pcm=read(rs,"short_pcm",{801,2}),short_enc=read(rs,"short_encoded",{2,32,2});
    cudaDeviceProp prop;MX_CUDA(cudaGetDeviceProperties(&prop,0));std::printf("GPU0：%s\n",prop.name);
    int fails=0,passed=0,layouts=0;
    for(const auto& weight:{root/"video/minimax-h3/vae/minimax_h3_audio_vae_fp32.safetensors",
                           root/"video/minimax-h3/hf/audio_vae/diffusion_pytorch_model.safetensors"}) {
      if(!fs::exists(weight)){std::printf("SKIP：%s\n",weight.string().c_str());continue;}
      ++layouts;std::printf("權重：%s\n",weight.string().c_str());
      TensorStore st;st.add_file(weight.string());H3AudioVae vae(st,0);Ctx c(0,1<<20,1<<20);
      MX_BLAS(cublasSetMathMode(c.blas,CUBLAS_DEFAULT_MATH));
      DevBuf z(latent.size()*sizeof(float),0);MX_CUDA(cudaMemcpy(z.get(),latent.data(),z.size(),cudaMemcpyHostToDevice));
      std::vector<float> decoded;vae.decode(c,z.as<float>(),80,decoded);
      double error=relative(decoded,stereo),snr=-20*std::log10(std::max(error,1e-30));
      bool ok=snr>=30;passed+=ok;fails+=!ok;std::printf("decode SNR：%.4f dB → %s\n",snr,ok?"PASS":"FAIL");
      int64_t A=-1;vae.encode(c,pcm.data(),64000,z.as<float>(),A);
      std::vector<float> got(latent.size());MX_CUDA(cudaMemcpy(got.data(),z.get(),z.size(),cudaMemcpyDeviceToHost));
      error=relative(got,encoded);ok=A==80 && error<0.03;passed+=ok;fails+=!ok;
      std::printf("encode：A=%lld，相對誤差 %.6f → %s\n",static_cast<long long>(A),error,ok?"PASS":"FAIL");
      vae.encode(c,short_pcm.data(),801,z.as<float>(),A);got.resize(short_enc.size());
      MX_CUDA(cudaMemcpy(got.data(),z.get(),got.size()*sizeof(float),cudaMemcpyDeviceToHost));
      error=relative(got,short_enc);ok=A==2 && error<0.03;passed+=ok;fails+=!ok;
      std::printf("encode 尾端補零：A=%lld，相對誤差 %.6f → %s\n",static_cast<long long>(A),error,ok?"PASS":"FAIL");
      ok=rejects([&]{vae.decode(c,nullptr,80,decoded);}) && rejects([&]{vae.decode(c,z.as<float>(),0,decoded);}) &&
         rejects([&]{vae.encode(c,pcm.data(),0,z.as<float>(),A);});
      auto bad=short_pcm;bad[0]=std::numeric_limits<float>::quiet_NaN();
      ok=ok && rejects([&]{vae.encode(c,bad.data(),801,z.as<float>(),A);});
      passed+=ok;fails+=!ok;std::printf("無效輸入拒絕：%s\n",ok?"PASS":"FAIL");
      std::vector<float> long_z(2*32*400);
      for(int ch=0;ch<2;ch++)for(int d=0;d<32;d++)for(int t=0;t<400;t++)long_z[(ch*32+d)*400+t]=latent[(ch*32+d)*80+t%80];
      DevBuf bench(long_z.size()*sizeof(float),0);MX_CUDA(cudaMemcpy(bench.get(),long_z.data(),bench.size(),cudaMemcpyHostToDevice));
      std::vector<double> times;vae.decode(c,bench.as<float>(),400,decoded);
      for(int j=0;j<3;j++) {
        auto start=std::chrono::steady_clock::now();vae.decode(c,bench.as<float>(),400,decoded);
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
      }
      std::sort(times.begin(),times.end());ok=decoded.size()==640000;passed+=ok;fails+=!ok;
      std::printf("10 秒 decode：%.3f ms（3 次中位數，含 PCM 回傳），%zu floats → %s\n",times[1],decoded.size(),ok?"PASS":"FAIL");
    }
    std::printf("結果：%d layouts，%d PASS，%d FAIL\n",layouts,passed,fails);return fails?1:0;
  }catch(const std::exception& e){std::printf("FAIL：%s\n",e.what());return 1;}
}
