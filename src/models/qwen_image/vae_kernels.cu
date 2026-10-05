#include "models/qwen_image/vae_kernels.h"
#include <cmath>
#include <cstdint>

namespace mx::qvae {
namespace {
__device__ float rounded(float x) { return __bfloat162float(__float2bfloat16(x)); }
// T 為 bf16（逐元素）或 uint4（8 個通道一組）；ci 以 T 為單位。
template <class T>
__global__ void columns_kernel(const T* x, T* col, int h, int w, int ci, int ow,
                               int k, int stride, int pad, bool up, int start, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = i % ci, tap = (i / ci) % (k * k), p = start + i / (ci * k * k);
  int yy = (p / ow) * stride - pad + tap / k;
  int xx = (p % ow) * stride - pad + tap % k;
  int vh = up ? 2 * h : h, vw = up ? 2 * w : w;
  col[i] = (yy >= 0 && yy < vh && xx >= 0 && xx < vw)
      ? x[(int64_t(up ? yy / 2 : yy) * w + (up ? xx / 2 : xx)) * ci + c]
      : T{};
}
__global__ void finish_kernel(const float* x, const float* b, const bf16* r, bf16* y, int64_t n, int c) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // PyTorch 的 cudnn Conv2d 先輸出 bf16，再另外加 bias；殘差相加再捨入一次。
  float v = rounded(rounded(x[i]) + b[i % c]);
  if (r) v += __bfloat162float(r[i]);
  y[i] = __float2bfloat16(v);
}
// 每個 warp 處理一個像素（c 為偶數）。
__global__ void norm_kernel(const bf16* x, const float* g, bf16* y, int pixels, int c, bool silu) {
  int p = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (p >= pixels) return;
  const auto* in = reinterpret_cast<const __nv_bfloat162*>(x + int64_t(p) * c);
  auto* out = reinterpret_cast<__nv_bfloat162*>(y + int64_t(p) * c);
  float sum = 0.f;
  for (int j = lane; j < c / 2; j += 32) { float2 v = __bfloat1622float2(in[j]); sum += v.x*v.x + v.y*v.y; }
  for (int o = 16; o; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  float inv = 1.f / fmaxf(sqrtf(sum), 1e-12f), scale = sqrtf(float(c));
  for (int j = lane; j < c / 2; j += 32) {
    float2 v = __bfloat1622float2(in[j]);
    float r[2] = {v.x, v.y};
    for (int e = 0; e < 2; ++e) {
      // 對齊 diffusers：normalize、乘 sqrt(C)、乘 gamma 各自捨入 bf16。
      float t = rounded(r[e] * inv);
      t = rounded(rounded(t * scale) * g[2*j+e]);
      r[e] = silu ? t / (1.f + expf(-t)) : t;
    }
    out[j] = __floats2bfloat162_rn(r[0], r[1]);
  }
}
// (top, left)：輸出區域在「x 放大後」座標中的起點（分塊時輸出比 x 少一圈）。
__global__ void shortcut_kernel(bf16* y, const bf16* x, int w, int ci, int co, int ow,
                                int fs, int ft, bool up, int top, int left, int64_t n) {
  int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= n) return;
  int c = i % co, p = i/co, yy = p/ow + top, xx = p%ow + left;
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

// ---- implicit GEMM 卷積：Y[p, n] = Σ_k A[p, k]·W[n, k]，A 直接從 NHWC 輸入讀（不展開 im2col）。
// 區塊 256 像素 × 144 輸出通道 × 32 個 K；8 個 warp 排成 4×2，每個 warp 64×72。
constexpr int kBM = 256, kBN = 144, kBK = 32, kStages = 3, kThreads = 256;
constexpr int kMI = kBM / 64;  // 每個 warp 的 m16 數，也是每個執行緒負責的 A 列數
constexpr int kStageBytes = (kBM + kBN) * kBK * 2;
struct ConvArgs {
  const bf16* x; const bf16* w; const float* bias; const bf16* r; bf16* y;
  int h, wd, ci, oh, ow, co, k, stride, pady, padx, up, M, K;
  int rw, ry, rx;  // 殘差的寬度與輸出在殘差中的起點
};
// 每列 4 個 16 bytes 區塊，XOR 交錯避免 ldmatrix bank 衝突。
__device__ __forceinline__ int swz(int row, int chunk) { return (row * 4 + (chunk ^ ((row >> 1) & 3))) * 16; }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 一個 stage：A kBM 列、B 144 列，各 4 個 16 bytes 區塊；padding 與 K 尾端以 0 填。
__device__ __forceinline__ void load_stage(const ConvArgs& a, uint32_t sa, int kc, const int* oy,
                                           const int* ox, int m0, int n0) {
  int t = threadIdx.x, chunk = t % 4, k = kc * kBK + chunk * 8;
  bool kin = k < a.K;
  int tap = kin ? k / a.ci : 0, ch = k - tap * a.ci, ky = tap / a.k, kx = tap - ky * a.k;
  int vh = a.up ? 2 * a.h : a.h, vw = a.up ? 2 * a.wd : a.wd;
  #pragma unroll
  for (int i = 0; i < kMI; ++i) {
    int row = t / 4 + i * 64, yy = oy[i] + ky, xx = ox[i] + kx;
    bool ok = kin && m0 + row < a.M && yy >= 0 && yy < vh && xx >= 0 && xx < vw;
    const bf16* src = ok ? a.x + (int64_t(a.up ? yy >> 1 : yy) * a.wd + (a.up ? xx >> 1 : xx)) * a.ci + ch : a.x;
    cp16(sa + swz(row, chunk), src, ok);
  }
  uint32_t sb = sa + kBM * kBK * 2;
  for (int id = t; id < kBN * 4; id += kThreads) {
    int row = id / 4;
    cp16(sb + swz(row, chunk), kin ? a.w + int64_t(n0 + row) * a.K + k : a.w, kin);
  }
}
__global__ void __launch_bounds__(kThreads, 1) conv_kernel(ConvArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint32_t base = uint32_t(__cvta_generic_to_shared(smem));
  int t = threadIdx.x, lane = t % 32, warp = t / 32, wm = warp % 4, wn = warp / 4;
  int n0 = blockIdx.x * kBN, m0 = blockIdx.y * kBM;
  int oy[kMI], ox[kMI];
  #pragma unroll
  for (int i = 0; i < kMI; ++i) {
    int p = min(m0 + t / 4 + i * 64, a.M - 1);
    oy[i] = (p / a.ow) * a.stride - a.pady;
    ox[i] = (p % a.ow) * a.stride - a.padx;
  }
  float acc[kMI][9][4] = {};
  int nk = (a.K + kBK - 1) / kBK;
  #pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < nk) load_stage(a, base + s * kStageBytes, s, oy, ox, m0, n0);
    asm volatile("cp.async.commit_group;\n");
  }
  for (int kc = 0; kc < nk; ++kc) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
    __syncthreads();
    int next = kc + kStages - 1;
    if (next < nk) load_stage(a, base + (next % kStages) * kStageBytes, next, oy, ox, m0, n0);
    asm volatile("cp.async.commit_group;\n");
    uint32_t sa = base + (kc % kStages) * kStageBytes, sb = sa + kBM * kBK * 2;
    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t af[kMI][4], bfr[9][2];
      #pragma unroll
      for (int mi = 0; mi < kMI; ++mi)
        ldsm4(af[mi], sa + swz(wm * (kBM / 4) + mi * 16 + lane % 16, ks * 2 + lane / 16));
      #pragma unroll
      for (int ni = 0; ni < 8; ni += 2) {
        uint32_t r[4];
        ldsm4(r, sb + swz(wn * 72 + ni * 8 + (lane / 16) * 8 + lane % 8, ks * 2 + (lane / 8) % 2));
        bfr[ni][0] = r[0]; bfr[ni][1] = r[1]; bfr[ni + 1][0] = r[2]; bfr[ni + 1][1] = r[3];
      }
      ldsm2(bfr[8], sb + swz(wn * 72 + 64 + lane % 8, ks * 2 + (lane / 8) % 2));
      #pragma unroll
      for (int mi = 0; mi < kMI; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 9; ++ni) mma(acc[mi][ni], af[mi], bfr[ni]);
    }
  }
  asm volatile("cp.async.wait_group 0;\n");
  // 收尾同 finish_kernel：先捨入 bf16，再加 bias，再加殘差。
  #pragma unroll
  for (int mi = 0; mi < kMI; ++mi)
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      int p = m0 + wm * (kBM / 4) + mi * 16 + lane / 4 + half * 8;
      if (p >= a.M) continue;
      int py = p / a.ow;
      int64_t ro = (int64_t(py + a.ry) * a.rw + p - py * a.ow + a.rx) * a.co;
      #pragma unroll
      for (int ni = 0; ni < 9; ++ni) {
        int n = n0 + wn * 72 + ni * 8 + 2 * (lane % 4);
        int64_t o = int64_t(p) * a.co + n;
        float v0 = rounded(rounded(acc[mi][ni][half * 2]) + a.bias[n]);
        float v1 = rounded(rounded(acc[mi][ni][half * 2 + 1]) + a.bias[n + 1]);
        if (a.r) {
          float2 r = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a.r + ro + n));
          v0 += r.x; v1 += r.y;
        }
        *reinterpret_cast<__nv_bfloat162*>(a.y + o) = __floats2bfloat162_rn(v0, v1);
      }
    }
}
// 輸出只有 4 通道（conv_out）時 tensor core 用不滿：每個執行緒直接算一個像素，權重放 shared。
constexpr int kSmallCo = 4;
__global__ void conv_small_kernel(ConvArgs a) {
  extern __shared__ float4 wsm[];  // [K]，每個 K 存 4 個輸出通道
  for (int i = threadIdx.x; i < a.K; i += blockDim.x)
    wsm[i] = make_float4(__bfloat162float(a.w[i]), __bfloat162float(a.w[a.K + i]),
                         __bfloat162float(a.w[2 * a.K + i]), __bfloat162float(a.w[3 * a.K + i]));
  __syncthreads();
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= a.M) return;
  int oy = (p / a.ow) * a.stride - a.pady, ox = (p % a.ow) * a.stride - a.padx;
  int vh = a.up ? 2 * a.h : a.h, vw = a.up ? 2 * a.wd : a.wd;
  float acc[kSmallCo] = {};
  for (int tap = 0; tap < a.k * a.k; ++tap) {
    int yy = oy + tap / a.k, xx = ox + tap % a.k;
    if (yy < 0 || yy >= vh || xx < 0 || xx >= vw) continue;
    const auto* src = reinterpret_cast<const uint4*>(
        a.x + (int64_t(a.up ? yy >> 1 : yy) * a.wd + (a.up ? xx >> 1 : xx)) * a.ci);
    const float4* wt = wsm + tap * a.ci;
    for (int c = 0; c < a.ci / 8; ++c) {
      uint4 v = src[c];
      const auto* e = reinterpret_cast<const __nv_bfloat162*>(&v);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(e[j]);
        float4 w0 = wt[c * 8 + 2 * j], w1 = wt[c * 8 + 2 * j + 1];
        acc[0] += f.x * w0.x + f.y * w1.x; acc[1] += f.x * w0.y + f.y * w1.y;
        acc[2] += f.x * w0.z + f.y * w1.z; acc[3] += f.x * w0.w + f.y * w1.w;
      }
    }
  }
  #pragma unroll
  for (int n = 0; n < kSmallCo; ++n) {
    a.y[int64_t(p) * kSmallCo + n] = __float2bfloat16(rounded(rounded(acc[n]) + a.bias[n]));
  }
}
}  // namespace
bool conv_supported(int ci, int co) {
  return ci % 8 == 0 && (co % kBN == 0 || (co == kSmallCo && ci * 9 * 16 <= 48 * 1024));
}
void conv(const bf16* x, const bf16* w, const float* bias, const bf16* r, int rw, int ry, int rx, bf16* y,
          int h, int wd, int ci, int oh, int ow, int co, int k, int stride, int pady, int padx, bool up,
          cudaStream_t s) {
  MX_CHECK(conv_supported(ci, co) && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(w) % 16 == 0, "VAE implicit conv: unsupported shape or alignment");
  int64_t M = int64_t(oh) * ow;
  MX_CHECK(M > 0 && (M + kBM - 1) / kBM <= 65535, "VAE implicit conv: too many pixels");
  // 逐次設定（主機端很便宜），多 GPU 時不必追蹤哪張卡設過。
  MX_CUDA(cudaFuncSetAttribute(conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kStages * kStageBytes));
  ConvArgs a{x, w, bias, r, y, h, wd, ci, oh, ow, co, k, stride, pady, padx, up ? 1 : 0, int(M), ci * k * k,
             rw, ry, rx};
  if (co == kSmallCo) {
    MX_CHECK(!r, "VAE small conv: residual not supported");
    conv_small_kernel<<<unsigned((M + 255) / 256), 256, a.K * sizeof(float4), s>>>(a);
    MX_KCHECK();
    return;
  }
  conv_kernel<<<dim3(co / kBN, unsigned((M + kBM - 1) / kBM)), kThreads, kStages * kStageBytes, s>>>(a);
  MX_KCHECK();
}
void columns(const bf16* x,bf16* y,int h,int w,int c,int oh,int ow,int k,int st,int pad,bool up,int start,int rows,cudaStream_t s) {
  bool vec=c%8==0 && reinterpret_cast<uintptr_t>(x)%16==0 && reinterpret_cast<uintptr_t>(y)%16==0;
  int64_t n=int64_t(rows)*(vec?c/8:c)*k*k;
  MX_CHECK(n<INT32_MAX,"VAE im2col chunk too large");
  if(vec) columns_kernel<<<unsigned((n+255)/256),256,0,s>>>(reinterpret_cast<const uint4*>(x),
      reinterpret_cast<uint4*>(y),h,w,c/8,ow,k,st,pad,up,start,int(n));
  else columns_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,y,h,w,c,ow,k,st,pad,up,start,int(n));
  MX_KCHECK();
}
void finish(const float* x,const float* b,const bf16* r,bf16* y,int rows,int c,cudaStream_t s) {
  int64_t n=int64_t(rows)*c; finish_kernel<<<unsigned((n+255)/256),256,0,s>>>(x,b,r,y,n,c); MX_KCHECK();
}
void norm(const bf16* x,const float* g,bf16* y,int p,int c,bool si,cudaStream_t s) {
  MX_CHECK(c%2==0,"VAE norm channels must be even");
  norm_kernel<<<(p+7)/8,256,0,s>>>(x,g,y,p,c,si); MX_KCHECK();
}
void shortcut(bf16* y,int oh,int ow,const bf16* x,int w,int ci,int co,int fs,int ft,bool up,int top,int left,cudaStream_t s) {
  MX_CHECK(up || (top==0 && left==0),"VAE shortcut offset only supported when upsampling");
  int64_t n=int64_t(oh)*ow*co;
  shortcut_kernel<<<unsigned((n+255)/256),256,0,s>>>(y,x,w,ci,co,ow,fs,ft,up,top,left,n); MX_KCHECK();
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
