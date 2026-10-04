#include "models/minimax_h3/audio_vae_kernels.h"
#include <cmath>
namespace mx::h3audio {
namespace {
__global__ void columns(const float* x, float* y, int C, int T, int O, int K, int S, int P, int D) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= int64_t(C)*K*O) return;
  int t = int(i%O)*S-P+int(i/O%K)*D;
  y[i] = t>=0 && t<T ? x[(i/O/K)*T+t] : 0.f;
}
__global__ void scatter(const float* x, float* y, const float* b, int C, int T, int O, int K, int S, int P) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= int64_t(C)*O) return;
  int ch=int(i/O), t=int(i%O); float v=b?b[ch]:0.f;
  for (int k=0;k<K;k++) {
    int u=t+P-k;
    if (u>=0 && u%S==0 && u/S<T) v+=x[(int64_t(ch)*K+k)*T+u/S];
  }
  y[i]=v;
}
__global__ void addbias(float* x,const float* b,int C,int T) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<int64_t(C)*T) x[i]+=b[i/T];
}
__global__ void sn(const float* x,float* y,const float* a,int C,int T) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=int64_t(C)*T) return;
  float alpha=a[i/T], v=x[i], s=sinf(alpha*v);
  y[i]=v+s*s/(alpha+1e-9f);
}
// 與 replicate-pad -> depthwise transposed-conv -> crop 相同，直接算裁切後的位置。
__global__ void up(const float* x,float* y,const float* f,const float* a,const float* b,int C,int T,int K) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=int64_t(C)*T*2) return;
  int ch=int(i/(T*2)), t=int(i%(T*2)), pad=K/2-1, left=pad*2+(K-2)/2;
  float v=0.f;
  for(int k=0;k<K;k++) {
    int u=t+left-k;
    if(u>=0 && u%2==0 && u/2<T+2*pad) v+=2.f*f[k]*x[int64_t(ch)*T+min(max(u/2-pad,0),T-1)];
  }
  float alpha=expf(a[ch]), beta=expf(b[ch]), s=sinf(alpha*v);
  y[i]=v+s*s/(beta+1e-9f);
}
__global__ void down(const float* x,float* y,const float* f,int C,int T,int K) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=int64_t(C)*T) return;
  int ch=int(i/T), t=int(i%T); float v=0.f;
  for(int k=0;k<K;k++) v+=f[k]*x[int64_t(ch)*T*2+min(max(2*t-(K/2-1)+k,0),2*T-1)];
  y[i]=v;
}
__global__ void tr(const float* x,float* y,int C,int T) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<int64_t(C)*T) y[(i%T)*C+i/T]=x[i];
}
__global__ void ln(const float* x,float* y,const float* w,const float* b,int D) {
  __shared__ float sums[256], vars[256];
  int r=blockIdx.x, lane=threadIdx.x; float s=0.f;
  for(int j=lane;j<D;j+=256) s+=x[int64_t(r)*D+j];
  sums[lane]=s; __syncthreads();
  for(int k=128;k;k/=2){if(lane<k)sums[lane]+=sums[lane+k];__syncthreads();}
  float mean=sums[0]/D; s=0.f;
  for(int j=lane;j<D;j+=256){float v=x[int64_t(r)*D+j]-mean;s+=v*v;}
  vars[lane]=s; __syncthreads();
  for(int k=128;k;k/=2){if(lane<k)vars[lane]+=vars[lane+k];__syncthreads();}
  float scale=rsqrtf(vars[0]/D+1e-5f);
  for(int j=lane;j<D;j+=256)y[int64_t(r)*D+j]=(x[int64_t(r)*D+j]-mean)*scale*w[j]+b[j];
}
__global__ void gg(float* x,const float* y,int64_t n) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n){float v=x[i];x[i]=0.5f*v*(1.f+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)))*y[i];}
}
__global__ void sm(float* x,int T,int D) {
  __shared__ float a[256];
  int r=blockIdx.x,j=threadIdx.x;float m=-INFINITY;
  for(int k=j;k<=r;k+=256)m=fmaxf(m,x[r*T+k]/sqrtf(float(D)));
  a[j]=m;__syncthreads();
  for(int k=128;k;k/=2){if(j<k)a[j]=fmaxf(a[j],a[j+k]);__syncthreads();}
  m=a[0];float s=0.f;
  for(int k=j;k<T;k+=256){float v=k<=r?expf(x[r*T+k]/sqrtf(float(D))-m):0.f;x[r*T+k]=v;s+=v;}
  a[j]=s;__syncthreads();
  for(int k=128;k;k/=2){if(j<k)a[j]+=a[j+k];__syncthreads();}
  for(int k=j;k<T;k+=256)x[r*T+k]/=a[0];
}
__global__ void avg(const float* x,float* y,int T,int H,int D,int O) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=T*O)return;
  int t=i/O,o=i%O,start=o*D/O,end=((o+1)*D+O-1)/O;float v=0.f;
  for(int d=start;d<end;d++){float hsum=0.f;for(int h=0;h<H;h++)hsum+=x[(int64_t(h)*T+t)*D+d];v+=hsum/H;}
  y[i]=v/(end-start);
}
__global__ void nz(const float* x,float* y,const float* mean,const float* std,int T,bool inverse) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=32*T)return;
  y[i]=inverse?(x[i]-mean[i/T])/std[i/T]:x[i]*std[i/T]+mean[i/T];
}
int blocks(int64_t n){return int((n+255)/256);}
}
void im2col(const float* x,float* y,int C,int T,int O,int K,int S,int P,int D,cudaStream_t s){columns<<<blocks(int64_t(C)*K*O),256,0,s>>>(x,y,C,T,O,K,S,P,D);MX_KCHECK();}
void col2im(const float* x,float* y,const float* b,int C,int T,int O,int K,int S,int P,cudaStream_t s){scatter<<<blocks(int64_t(C)*O),256,0,s>>>(x,y,b,C,T,O,K,S,P);MX_KCHECK();}
void bias(float* x,const float* b,int C,int T,cudaStream_t s){if(b){addbias<<<blocks(int64_t(C)*T),256,0,s>>>(x,b,C,T);MX_KCHECK();}}
void snake(const float* x,float* y,const float* a,int C,int T,cudaStream_t s){sn<<<blocks(int64_t(C)*T),256,0,s>>>(x,y,a,C,T);MX_KCHECK();}
void alias_up(const float* x,float* y,const float* f,const float* a,const float* b,int C,int T,int K,cudaStream_t s){up<<<blocks(int64_t(C)*T*2),256,0,s>>>(x,y,f,a,b,C,T,K);MX_KCHECK();}
void alias_down(const float* x,float* y,const float* f,int C,int T,int K,cudaStream_t s){down<<<blocks(int64_t(C)*T),256,0,s>>>(x,y,f,C,T,K);MX_KCHECK();}
void transpose(const float* x,float* y,int C,int T,cudaStream_t s){tr<<<blocks(int64_t(C)*T),256,0,s>>>(x,y,C,T);MX_KCHECK();}
void norm(const float* x,float* y,const float* w,const float* b,int T,int D,cudaStream_t s){ln<<<T,256,0,s>>>(x,y,w,b,D);MX_KCHECK();}
void geglu(float* x,const float* y,int64_t n,cudaStream_t s){gg<<<blocks(n),256,0,s>>>(x,y,n);MX_KCHECK();}
void softmax(float* x,int T,int D,cudaStream_t s){sm<<<T,256,0,s>>>(x,T,D);MX_KCHECK();}
void pool(const float* x,float* y,int T,int H,int D,int O,cudaStream_t s){avg<<<blocks(int64_t(T)*O),256,0,s>>>(x,y,T,H,D,O);MX_KCHECK();}
void normalize(const float* x,float* y,const float* m,const float* v,int T,bool inverse,cudaStream_t s){nz<<<blocks(int64_t(32)*T),256,0,s>>>(x,y,m,v,T,inverse);MX_KCHECK();}
}
