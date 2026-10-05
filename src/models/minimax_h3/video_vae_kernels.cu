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
__global__ void swiglu_kernel(const bf16* gu, bf16* out, const float* bias, int f, bool first) {
  int col = (blockIdx.y*blockDim.x+threadIdx.x)*8;
  if (col >= f) return;
  int pos = blockIdx.x*2*f+col, gate_col=col+(first?0:f), up_col=col+(first?f:0);
  alignas(16) bf16 gate[8],up[8],result[8];
  *reinterpret_cast<uint4*>(gate)=*reinterpret_cast<const uint4*>(gu+pos+(first?0:f));
  *reinterpret_cast<uint4*>(up)=*reinterpret_cast<const uint4*>(gu+pos+(first?f:0));
  #pragma unroll
  for(int j=0;j<8;++j) {
    // 保留 GEMM → bias → SiLU → 乘法各自寫回 bf16 的取整順序。
    float g=__bfloat162float(__float2bfloat16(__bfloat162float(gate[j])+bias[gate_col+j]));
    float u=__bfloat162float(__float2bfloat16(__bfloat162float(up[j])+bias[up_col+j]));
    float act=__bfloat162float(__float2bfloat16(g/(1.f+expf(-g))));
    result[j]=__float2bfloat16(act*u);
  }
  *reinterpret_cast<uint4*>(out+blockIdx.x*f+col)=*reinterpret_cast<const uint4*>(result);
}
__global__ void biased_residual_kernel(float* x,const bf16* y,const float* bias,const float* scale,int dim) {
  int col=(blockIdx.y*blockDim.x+threadIdx.x)*4;
  if(col>=dim)return;
  int pos=blockIdx.x*dim+col;
  alignas(8) bf16 values[4];
  *reinterpret_cast<uint2*>(values)=*reinterpret_cast<const uint2*>(y+pos);
  float4 result=*reinterpret_cast<const float4*>(x+pos);
  const float4 b=*reinterpret_cast<const float4*>(bias+col),s=*reinterpret_cast<const float4*>(scale+col);
  float* r=reinterpret_cast<float*>(&result);
  const float* bs=reinterpret_cast<const float*>(&b),*sc=reinterpret_cast<const float*>(&s);
  #pragma unroll
  for(int j=0;j<4;++j) {
    float v=__bfloat162float(__float2bfloat16(__bfloat162float(values[j])+bs[j]));
    v*=sc[j];r[j]+=v;
  }
  *reinterpret_cast<float4*>(x+pos)=result;
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
// Each col row holds kh*kw taps in order, each tap C contiguous channels: k = (ky*kw + kx)*C + c.
// With C % 8 == 0 a thread moves 8 bf16 (16 bytes) at once.
__global__ void im2col_kernel(const bf16* x, bf16* col, int64_t ldcol, int H, int W, int C,
    int kh, int kw, int stride, int pad, int OH, int OW, int start, int rows, int vec) {
  const int cv=C/vec, taps=kh*kw;
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=int64_t(rows)*taps*cv) return;
  int c=int(i%cv)*vec; int64_t rest=i/cv;
  int tap=int(rest%taps), r=int(rest/taps), pos=start+r;
  int ky=tap/kw, kx=tap%kw;
  int yy=reflect(pos/OW*stride-pad+ky,H), xx=reflect(pos%OW*stride-pad+kx,W);
  const bf16* src=x+(int64_t(yy)*W+xx)*C+c;
  bf16* dst=col+int64_t(r)*ldcol+int64_t(tap)*C+c;
  if (vec==8) *reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(src);
  else *dst=*src;
}
void launch_im2col(const bf16* x,bf16* col,int64_t ldcol,int H,int W,int C,int kh,int kw,int stride,int pad,
    int OH,int OW,int start,int rows,cudaStream_t s) {
  const int vec=C%8==0?8:1;
  const int64_t n=int64_t(rows)*kh*kw*(C/vec);
  im2col_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,col,ldcol,H,W,C,kh,kw,stride,pad,OH,OW,start,rows,vec); MX_KCHECK();
}
// blockDim must be a multiple of 32 and at most 1024.
__device__ float sum_block(float v) {
  __shared__ float sums[32];
  __syncthreads();  // a previous call may still be reading sums[0]
  for (int off=16;off;off/=2) v+=__shfl_down_sync(0xffffffff,v,off);
  if (!(threadIdx.x%32)) sums[threadIdx.x/32]=v;
  __syncthreads();
  v=threadIdx.x<blockDim.x/32?sums[threadIdx.x]:0;
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
void swiglu(const bf16* gu,bf16* out,const float* bias,int rows,int f,bool first,cudaStream_t s) {
  MX_CHECK(gu && out && bias && rows>0 && f>0 && f%8==0,"swiglu: invalid shape or pointer");
  swiglu_kernel<<<dim3(rows,(f/8+255)/256),256,0,s>>>(gu,out,bias,f,first); MX_KCHECK();
}
void biased_residual(float* x,const bf16* y,const float* bias,const float* scale,int rows,int dim,cudaStream_t s) {
  MX_CHECK(x && y && bias && scale && rows>0 && dim>0 && dim%4==0,"biased_residual: invalid shape or pointer");
  biased_residual_kernel<<<dim3(rows,(dim/4+255)/256),256,0,s>>>(x,y,bias,scale,dim); MX_KCHECK();
}
void unpack(const bf16* a,float* out,int T,int H,int W,int pt,int p,cudaStream_t s) {
  unpack_kernel<<<unsigned((int64_t(T)*pt*H*p*W*p*3+255)/256),256,0,s>>>(a,out,T,H,W,pt,p); MX_KCHECK();
}
void image_input(const uint8_t* rgb,bf16* out,int H,int W,int y,int x,int h,int w,cudaStream_t s) {
  image_kernel<<<(h*w*3+255)/256,256,0,s>>>(rgb,out,H,W,y,x,h,w); MX_KCHECK();
}
void im2col(const bf16* x,bf16* col,int H,int W,int C,int kh,int kw,int stride,int pad,
    int OH,int OW,int start,int rows,cudaStream_t s) {
  launch_im2col(x,col,int64_t(C)*kh*kw,H,W,C,kh,kw,stride,pad,OH,OW,start,rows,s);
}
void im2col_ld(const bf16* x,bf16* col,int64_t ldcol,int H,int W,int C,int kh,int kw,int stride,int pad,
    int OH,int OW,int start,int rows,cudaStream_t s) {
  launch_im2col(x,col,ldcol,H,W,C,kh,kw,stride,pad,OH,OW,start,rows,s);
}
void groupnorm_silu(const bf16* x,bf16* out,const float* w,const float* b,int spatial,int C,int groups,float eps,cudaStream_t s) {
  gn_kernel<<<groups,1024,0,s>>>(x,out,w,b,spatial,C,groups,eps); MX_KCHECK();
}
void channel_normalize(const float* m,float* z,const float* mean,const float* std,int rows,cudaStream_t s) {
  normalize_kernel<<<(rows*24+255)/256,256,0,s>>>(m,z,mean,std,rows); MX_KCHECK();
}
}  // namespace mx::h3v
