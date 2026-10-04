#include "models/qwen_image/vae_kernels.h"
#include <cmath>

namespace mx::qvae {
namespace {
__device__ float rounded(float x) { return __bfloat162float(__float2bfloat16(x)); }
__global__ void columns_kernel(const bf16* x, bf16* col, int h, int w, int ci, int ow,
                               int k, int stride, int pad, bool up, int start, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = i % ci, tap = (i / ci) % (k * k), p = start + i / (ci * k * k);
  int yy = (p / ow) * stride - pad + tap / k;
  int xx = (p % ow) * stride - pad + tap % k;
  int vh = up ? 2 * h : h, vw = up ? 2 * w : w;
  col[i] = (yy >= 0 && yy < vh && xx >= 0 && xx < vw)
      ? x[(int64_t(up ? yy / 2 : yy) * w + (up ? xx / 2 : xx)) * ci + c]
      : __float2bfloat16(0.f);
}
__global__ void finish_kernel(const float* x, const float* b, bf16* y, int64_t n, int c) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // PyTorch 的 cudnn Conv2d 先輸出 bf16，再另外加 bias。
  if (i < n) y[i] = __float2bfloat16(rounded(x[i]) + b[i % c]);
}
__global__ void norm_kernel(const bf16* x, const float* g, bf16* y, int c, bool silu) {
  __shared__ float sums[256];
  int p = blockIdx.x, t = threadIdx.x;
  float sum = 0.f;
  for (int j = t; j < c; j += 256) { float v = __bfloat162float(x[int64_t(p)*c+j]); sum += v*v; }
  sums[t] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) { if (t < d) sums[t] += sums[t+d]; __syncthreads(); }
  float inv = 1.f / fmaxf(sqrtf(sums[0]), 1e-12f), scale = sqrtf(float(c));
  for (int j = t; j < c; j += 256) {
    // 對齊 diffusers：normalize、乘 sqrt(C)、乘 gamma 各自捨入 bf16。
    float v = rounded(__bfloat162float(x[int64_t(p)*c+j]) * inv);
    v = rounded(rounded(v * scale) * g[j]);
    if (silu) v = v / (1.f + expf(-v));
    y[int64_t(p)*c+j] = __float2bfloat16(v);
  }
}
__global__ void add_kernel(bf16* y, const bf16* x, int64_t n) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(__bfloat162float(y[i])+__bfloat162float(x[i]));
}
__global__ void shortcut_kernel(bf16* y, const bf16* x, int h, int w, int ci, int co,
                                int fs, int ft, bool up, int64_t n) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= n) return;
  int c = i % co, ow = up ? w*fs : w/fs, p = i/co, yy = p/ow, xx = p%ow;
  float v = 0;
  if (up) {
    int repeats = co*ft*fs*fs/ci;
    int ic = (c*ft*fs*fs+(ft-1)*fs*fs+(yy%fs)*fs+xx%fs)/repeats;
    v = __bfloat162float(x[(int64_t(yy/fs)*w+xx/fs)*ci+ic]);
  } else {
    int group = ci*ft*fs*fs/co;
    for (int j=0; j<group; ++j) {
      int a = c*group+j, ic = a/(ft*fs*fs), t = (a/(fs*fs))%ft;
      if (t != ft-1) continue;  // 單幀的時間左側補零。
      int sy = yy*fs+(a/fs)%fs, sx = xx*fs+a%fs;
      v += __bfloat162float(x[(int64_t(sy)*w+sx)*ci+ic]);
    }
    v = rounded(v/group);
  }
  y[i] = __float2bfloat16(__bfloat162float(y[i])+v);
}
__global__ void softmax_kernel(float* x, int c) {
  __shared__ float red[256];
  int t = threadIdx.x; int64_t base = int64_t(blockIdx.x)*c;
  float m = -INFINITY;
  for (int j=t;j<c;j+=256) m = fmaxf(m,x[base+j]);
  red[t]=m; __syncthreads();
  for(int d=128;d;d/=2) { if(t<d) red[t]=fmaxf(red[t],red[t+d]); __syncthreads(); }
  m=red[0]; float sum=0;
  for(int j=t;j<c;j+=256) sum+=expf(x[base+j]-m);
  red[t]=sum; __syncthreads();
  for(int d=128;d;d/=2) { if(t<d) red[t]+=red[t+d]; __syncthreads(); }
  sum=red[0];
  for(int j=t;j<c;j+=256) x[base+j]=expf(x[base+j]-m)/sum;
}
__global__ void value_kernel(const bf16* x,float* y,int c,int64_t n) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) y[i]=__bfloat162float(x[(i/c)*3*c+2*c+i%c]);
}
__global__ void attention_out_kernel(const float* x,bf16* y,int64_t n) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) y[i]=__float2bfloat16(x[i]);
}
__global__ void latent_in_kernel(const float* x, bf16* y, int p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<p*64) y[i]=__float2bfloat16(x[(i%64)*p+i/64]);
}
__global__ void latent_out_kernel(const bf16* x, float* y, int p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<p*64) y[i]=__bfloat162float(x[(i%p)*128+i/p]);
}
__global__ void image_in_kernel(const uint8_t* x, bf16* y, int p) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<p*4) y[i]=__float2bfloat16(float(x[i])*(2.f/255.f)-1.f);
}
__global__ void image_out_kernel(const bf16* x, uint8_t* y, int w, int fw, int top, int left,
                                int ch, int cw, int dt, int dl) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=ch*cw*4) return;
  int c=i%4,p=i/4, yy=p/cw,xx=p%cw;
  float v=__bfloat162float(x[(int64_t(yy+top)*w+xx+left)*4+c]);
  y[(int64_t(yy+dt)*fw+xx+dl)*4+c]=uint8_t(floorf(fminf(1.f,fmaxf(0.f,(v+1.f)*.5f))*255.f+.5f));
}
__global__ void crop_kernel(const bf16* x,bf16* y,int w,int c,int top,int left,int cw,int64_t n) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) { int64_t p=i/c; y[i]=x[((p/cw+top)*w+p%cw+left)*c+i%c]; }
}
}  // namespace
void columns(const bf16* x,bf16* y,int h,int w,int c,int oh,int ow,int k,int st,int pad,bool up,int start,int rows,cudaStream_t s) {
  int64_t n=int64_t(rows)*c*k*k;
  columns_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,y,h,w,c,ow,k,st,pad,up,start,n); MX_KCHECK();
}
void finish(const float* x,const float* b,bf16* y,int r,int c,cudaStream_t s) {
  int64_t n=int64_t(r)*c; finish_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,b,y,n,c); MX_KCHECK();
}
void norm(const bf16* x,const float* g,bf16* y,int p,int c,bool si,cudaStream_t s) {
  norm_kernel<<<p,256,0,s>>>(x,g,y,c,si); MX_KCHECK();
}
void add(bf16* y,const bf16* x,int64_t n,cudaStream_t s) {
  add_kernel<<<unsigned((n+255)/256),256,0,s>>>(y,x,n); MX_KCHECK();
}
void shortcut(bf16* y,const bf16* x,int h,int w,int ci,int co,int fs,int ft,bool up,cudaStream_t s) {
  int64_t n=int64_t(up?h*fs:h/fs)*(up?w*fs:w/fs)*co;
  shortcut_kernel<<<unsigned((n+255)/256),256,0,s>>>(y,x,h,w,ci,co,fs,ft,up,n); MX_KCHECK();
}
void softmax(float* x,int r,int c,cudaStream_t s) { softmax_kernel<<<r,256,0,s>>>(x,c); MX_KCHECK(); }
void attention_value(const bf16* x,float* y,int p,int c,cudaStream_t s) {
  int64_t n=int64_t(p)*c; value_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,y,c,n); MX_KCHECK();
}
void attention_out(const float* x,bf16* y,int64_t n,cudaStream_t s) {
  attention_out_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,y,n); MX_KCHECK();
}
void latent_in(const float* x,bf16* y,int p,cudaStream_t s) { latent_in_kernel<<<(p*64+255)/256,256,0,s>>>(x,y,p); MX_KCHECK(); }
void latent_out(const bf16* x,float* y,int p,cudaStream_t s) { latent_out_kernel<<<(p*64+255)/256,256,0,s>>>(x,y,p); MX_KCHECK(); }
void image_in(const uint8_t* x,bf16* y,int p,cudaStream_t s) { image_in_kernel<<<(p*4+255)/256,256,0,s>>>(x,y,p); MX_KCHECK(); }
void image_out(const bf16* x,uint8_t* y,int h,int w,int fw,int top,int left,int ch,int cw,int dt,int dl,cudaStream_t s) {
  image_out_kernel<<<(ch*cw*4+255)/256,256,0,s>>>(x,y,w,fw,top,left,ch,cw,dt,dl); MX_KCHECK();
}
void crop(const bf16* x,bf16* y,int w,int c,int top,int left,int h,int cw,cudaStream_t s) {
  int64_t n=int64_t(h)*cw*c; crop_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,y,w,c,top,left,cw,n); MX_KCHECK();
}
}  // namespace mx::qvae
