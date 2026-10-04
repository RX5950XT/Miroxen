#include "models/minimax_h3/video_vae_kernels.h"
#include <cmath>

namespace mx::h3v {
namespace {
__global__ void gather_kernel(const float* z, bf16* out, int F, int H, int W,
    int start, int T, int y, int x, int h, int w, const float* mean, const float* std) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T*h*w*24) return;
  int c = i%24, pos = i/24, xx = pos%w, yy = pos/w%h, t = min(start+pos/(h*w), F-1);
  out[i] = __float2bfloat16(z[((int64_t(c)*F+t)*H+y+yy)*W+x+xx]*std[c]+mean[c]);
}
__global__ void anchors_kernel(const bf16* p, float* x, const float* r, int N, int D, int nr) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < (N+nr+1)*D) x[i] = i < N*D ? __bfloat162float(p[i]) :
      (i < (N+nr)*D ? r[i-N*D] : 0.f);
}
__global__ void rope_kernel(float* cs, float* sn, int T, int H, int W, int suffix, int rot, float theta) {
  int i = blockIdx.x*blockDim.x+threadIdx.x, half = rot/2, axisdim = rot/6;
  if (i >= (T*H*W+suffix)*half) return;
  int tok = i/half, d = i%half, axis = d/axisdim;
  float pos = 0;
  if (tok < T*H*W) {
    int coords[] = {tok/(H*W), tok/W%H, tok%W};
    int sizes[] = {T,H,W};
    pos = 2.f*((coords[axis]+0.5f)/sizes[axis])-1.f;
  }
  float a = (6.28318530717958647692f*pos)*powf(theta, -float(d%axisdim)/axisdim);
  // 參考會把旋轉角轉成計算型別。
  cs[i] = __bfloat162float(__float2bfloat16(cosf(a)));
  sn[i] = __bfloat162float(__float2bfloat16(sinf(a)));
}
__global__ void swiglu_kernel(const bf16* gu, bf16* out, int rows, int f, bool first) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= rows*f) return;
  int pos = (i/f)*2*f+i%f;
  float gate = __bfloat162float(gu[pos+(first?0:f)]);
  float up = __bfloat162float(gu[pos+(first?f:0)]);
  // PyTorch SiLU 先寫回 bf16，再做乘法。
  float act = __bfloat162float(__float2bfloat16(gate/(1.f+expf(-gate))));
  out[i] = __float2bfloat16(act*up);
}
__global__ void unpack_kernel(const bf16* a, float* out, int T, int H, int W, int pt, int p) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int OH=H*p, OW=W*p;
  if (i >= int64_t(T)*pt*OH*OW*3) return;
  int c=i%3, xx=i/3%OW, yy=i/(3*OW)%OH, t=i/(int64_t(3)*OH*OW);
  int64_t tok = ((int64_t(t/pt)*H+yy/p)*W+xx/p);
  int col = ((c*pt+t%pt)*p+yy%p)*p+xx%p;
  out[i] = __bfloat162float(a[tok*(3*pt*p*p)+col]);
}
__global__ void image_kernel(const uint8_t* rgb, bf16* out, int H, int W,
    int y, int x, int h, int w) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=h*w*3) return;
  const float mean[]={0.485f,0.456f,0.406f}, std[]={0.229f,0.224f,0.225f};
  int c=i%3, pos=i/3;
  out[i]=__float2bfloat16((rgb[((y+pos/w)*W+x+pos%w)*3+c]/255.f-mean[c])/std[c]);
}
__device__ int reflect(int p, int n) { return p<0 ? -p : (p>=n ? 2*n-p-2 : p); }
__global__ void im2col_kernel(const bf16* x, bf16* col, int H, int W, int C,
    int kh, int kw, int stride, int pad, int OH, int OW, int start, int rows) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x, K=int64_t(C)*kh*kw;
  if (i>=rows*K) return;
  int k=i%K, pos=start+i/K;
  int c=k/(kh*kw), ky=k/kw%kh, kx=k%kw;
  int yy=reflect(pos/OW*stride-pad+ky,H), xx=reflect(pos%OW*stride-pad+kx,W);
  col[i]=x[(yy*W+xx)*C+c];
}
__device__ float sum_block(float v) {
  __shared__ float sums[8];
  for (int off=16;off;off/=2) v+=__shfl_down_sync(0xffffffff,v,off);
  if (!(threadIdx.x%32)) sums[threadIdx.x/32]=v;
  __syncthreads();
  v=threadIdx.x<8?sums[threadIdx.x]:0;
  if (threadIdx.x<32) for(int off=16;off;off/=2) v+=__shfl_down_sync(0xffffffff,v,off);
  if (!threadIdx.x) sums[0]=v;
  __syncthreads();
  return sums[0];
}
__global__ void gn_kernel(const bf16* x, bf16* out, const float* w, const float* b,
    int spatial, int C, int groups, float eps) {
  int cg=C/groups, count=spatial*cg, group=blockIdx.x;
  float sum=0;
  for(int i=threadIdx.x;i<count;i+=blockDim.x) sum+=__bfloat162float(x[(i/cg)*C+group*cg+i%cg]);
  float mean=sum_block(sum)/count;
  float variance=0;
  for(int i=threadIdx.x;i<count;i+=blockDim.x) {
    float d=__bfloat162float(x[(i/cg)*C+group*cg+i%cg])-mean; variance+=d*d;
  }
  float inv=rsqrtf(sum_block(variance)/count+eps);
  for(int i=threadIdx.x;i<count;i+=blockDim.x) {
    int c=group*cg+i%cg, pos=(i/cg)*C+c;
    float v=(__bfloat162float(x[pos])-mean)*inv*w[c]+b[c];
    out[pos]=__float2bfloat16(v/(1.f+expf(-v)));
  }
}
__global__ void normalize_kernel(const float* m, float* z, const float* mean, const float* std, int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<rows*24) {int c=i/rows, p=i%rows; z[i]=(m[p*48+c]-mean[c])/std[c];}
}
}  // namespace
void gather_latent(const float* z,bf16* out,int F,int H,int W,int start,int T,int y,int x,int h,int w,
    const float* mean,const float* std,cudaStream_t s) {
  gather_kernel<<<(T*h*w*24+255)/256,256,0,s>>>(z,out,F,H,W,start,T,y,x,h,w,mean,std); MX_KCHECK();
}
void anchors(const bf16* p,float* x,const float* r,int N,int D,int nr,cudaStream_t s) {
  anchors_kernel<<<((N+nr+1)*D+255)/256,256,0,s>>>(p,x,r,N,D,nr); MX_KCHECK();
}
void rope(float* cs,float* sn,int T,int H,int W,int suffix,int rot,float theta,cudaStream_t s) {
  rope_kernel<<<((T*H*W+suffix)*(rot/2)+255)/256,256,0,s>>>(cs,sn,T,H,W,suffix,rot,theta); MX_KCHECK();
}
void swiglu(const bf16* gu,bf16* out,int rows,int f,bool first,cudaStream_t s) {
  swiglu_kernel<<<(rows*f+255)/256,256,0,s>>>(gu,out,rows,f,first); MX_KCHECK();
}
void unpack(const bf16* a,float* out,int T,int H,int W,int pt,int p,cudaStream_t s) {
  unpack_kernel<<<unsigned((int64_t(T)*pt*H*p*W*p*3+255)/256),256,0,s>>>(a,out,T,H,W,pt,p); MX_KCHECK();
}
void image_input(const uint8_t* rgb,bf16* out,int H,int W,int y,int x,int h,int w,cudaStream_t s) {
  image_kernel<<<(h*w*3+255)/256,256,0,s>>>(rgb,out,H,W,y,x,h,w); MX_KCHECK();
}
void im2col(const bf16* x,bf16* col,int H,int W,int C,int kh,int kw,int stride,int pad,
    int OH,int OW,int start,int rows,cudaStream_t s) {
  im2col_kernel<<<unsigned((int64_t(rows)*C*kh*kw+255)/256),256,0,s>>>(x,col,H,W,C,kh,kw,stride,pad,OH,OW,start,rows); MX_KCHECK();
}
void groupnorm_silu(const bf16* x,bf16* out,const float* w,const float* b,int spatial,int C,int groups,float eps,cudaStream_t s) {
  gn_kernel<<<groups,256,0,s>>>(x,out,w,b,spatial,C,groups,eps); MX_KCHECK();
}
void channel_normalize(const float* m,float* z,const float* mean,const float* std,int rows,cudaStream_t s) {
  normalize_kernel<<<(rows*24+255)/256,256,0,s>>>(m,z,mean,std,rows); MX_KCHECK();
}
}  // namespace mx::h3v
