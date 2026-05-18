#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mma.h>

using namespace nvcuda; 

namespace OwnTensor {
    namespace autograd {
        #define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS (WARP_SIZE * WARPS_PER_BLOCK)

// -------------------- type helpers --------------------
template<typename T> __device__ inline float to_float(T x);
template<> __device__ inline float to_float(__half x) { return __half2float(x); }
template<> __device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T> __device__ __forceinline__ void store_v2(T* dst, float v0, float v1);
template<> __device__ __forceinline__ void store_v2<__half>(__half* dst, float v0, float v1) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
}
template<> __device__ __forceinline__ void store_v2<__nv_bfloat16>(__nv_bfloat16* dst, float v0, float v1) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
}

// -------------------- cp.async --------------------
__device__ __forceinline__ void cp_async16(void* __restrict__ dst, const void* __restrict__ src) {
    uint32_t smem_addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n":: "r"(smem_addr), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// -------------------- tile loader --------------------
template<typename T, int NUM_THREADS, int BYTES_PER_THREAD>
__device__ __forceinline__ void load_chunk_async(T* dst, const T* src, int total_bytes) {
    int total_vecs = total_bytes / 16;
    #pragma unroll
    for (int vi = threadIdx.x; vi < total_vecs; vi += NUM_THREADS)
        cp_async16(dst + (vi * 16 / sizeof(T)), src + (vi * 16 / sizeof(T)));
}

// Strided cp.async loader: copies the 16-byte-aligned portion of each row.
// Source stride is d (elements), dest stride is stride_smem (elements).
template<typename T, int NUM_THREADS>
__device__ __forceinline__ void load_chunk_async_strided(T* dst, const T* src, int rows, int d, int stride_smem) {
    constexpr int elems_per_vec = 16 / sizeof(T);       // 8 for fp16, 8 for bf16
    int vecs_per_row = d / elems_per_vec;                // full 16-byte vectors per row
    int total_vecs = rows * vecs_per_row;
    #pragma unroll
    for (int vi = threadIdx.x; vi < total_vecs; vi += NUM_THREADS) {
        int r = vi / vecs_per_row;
        int c_vec = vi % vecs_per_row;
        int c_elem = c_vec * elems_per_vec;
        cp_async16(dst + r * stride_smem + c_elem, src + r * d + c_elem);
    }
}

// Regular (non-async) strided load. Used when d is not a multiple of 8
// (for fp16) since cp.async requires 16-byte aligned source addresses and
// row starts at src + r*d won't be aligned when d % 8 != 0.
template<typename T, int NUM_THREADS>
__device__ __forceinline__ void load_strided_regular(T* dst, const T* src, int rows, int d, int stride_smem) {
    int total = rows * d;
    for (int i = threadIdx.x; i < total; i += NUM_THREADS) {
        int r = i / d;
        int c = i % d;
        dst[r * stride_smem + c] = src[i];
    }
}

// Cooperative SMEM zero-fill.
template<typename T, int NUM_THREADS>
__device__ __forceinline__ void zero_smem(T* dst, int count) {
    for (int i = threadIdx.x; i < count; i += NUM_THREADS)
        dst[i] = T(0);
}

// -------------------- FLASH ATTENTION --------------------
// Optimized: 4 warps, register-level softmax, direct register→global store,
// no float scratch buffer (S_PV_all eliminated)
template<typename T, typename WMMA_T, int Bc, int Br, int D_TILE = 128>
__global__ __launch_bounds__(THREADS)
void flashattn_wmma_kernel(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    T* __restrict__ O, float* __restrict__ L,
    int N, int d, int H, bool causal, float dropout_p, uint32_t dk0, uint32_t dk1
){
    extern __shared__ uint8_t smem_u8[];
    // d_padded must be >= round_up(d, 16) so WMMA tiles don't read across rows,
    // plus bank-conflict padding. Minimum 24 to fit a 16-wide WMMA tile + padding.
    const int d_padded = max(((d + 15) / 16) * 16 + 8, 24);
    T* Qi = reinterpret_cast<T*>(smem_u8);
    T* Kj = Qi + Br * d_padded;
    T* Vj = Kj + Bc * d_padded;
    // Only need P tile per warp (half). Float scratch buffer eliminated.
    const int tile_stride = 16 + 8;
    T* S_P_all = reinterpret_cast<T*>(Vj + Bc * d_padded);

    const int warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int batch = blockIdx.z, head = blockIdx.y, i0 = blockIdx.x * Br;
    if (i0 >= N) return;

    T* S_P_warp = S_P_all + warp_id * 16 * tile_stride;

    // Zero-fill Qi SMEM so cols [d..d_padded-1] are zero for WMMA
    zero_smem<T, THREADS>(Qi, Br * d_padded);
    __syncthreads();

    // Load Qi: use cp.async when d is 8-aligned (16-byte aligned rows), else regular loads
    constexpr int elems_per_vec = 16 / sizeof(T); // 8 for fp16/bf16
    const bool d_aligned = (d % elems_per_vec == 0);
    const int qi_rows = min(Br, N - i0);
    const T* q_src = Q + (long long)(batch * H + head) * N * d + (long long)i0 * d;
    if (d_aligned) {
        load_chunk_async_strided<T, THREADS>(Qi, q_src, qi_rows, d, d_padded);
        cp_async_commit(); cp_async_wait<0>();
    } else {
        load_strided_regular<T, THREADS>(Qi, q_src, qi_rows, d, d_padded);
    }
    __syncthreads();

    const float scaleF = rsqrtf((float)d);
    const long long kv_base = (long long)(batch * H + head) * N * d;

    const int warp_row_start = warp_id * 16;
    const bool warp_active = (warp_row_start < Br && i0 + warp_row_start < N);

    T* out_ptr = O + (long long)(batch * H + head) * N * d + (long long)(i0 + min(warp_row_start, Br - 16)) * d;
    float* l_ptr = L + (long long)(batch * H + head) * N + i0 + min(warp_row_start, Br - 16);

    // sm_80 accumulator fragment mapping (16x16):
    //   x[0,1] → row lane/4,   cols (lane%4)*2, (lane%4)*2+1
    //   x[2,3] → row lane/4+8, cols (lane%4)*2, (lane%4)*2+1
    //   x[4,5] → row lane/4,   cols (lane%4)*2+8, (lane%4)*2+9
    //   x[6,7] → row lane/4+8, cols (lane%4)*2+8, (lane%4)*2+9
    const int r_idx_low = lane / 4;
    const int r_idx_high = r_idx_low + 8;

    // Cache Qi fragments in registers
    wmma::fragment<wmma::matrix_a, 16, 16, 16, WMMA_T, wmma::row_major> fQ[8];
    if (warp_active) {
        #pragma unroll
        for (int t = 0; t < 128; t += 16) {
            if (t < d) wmma::load_matrix_sync(fQ[t/16], Qi + warp_row_start * d_padded + t, d_padded);
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) wmma::fill_fragment(acc[k], 0.f);

    float mi_low = -INFINITY, li_low = 0.f;
    float mi_high = -INFINITY, li_high = 0.f;

    for (int j0 = 0; j0 < N; j0 += Bc) {
        if (causal && j0 > i0 + Br - 1) break;

        const int kv_rows = min(Bc, N - j0);
        // Zero K/V SMEM so padding columns are zero for WMMA
        zero_smem<T, THREADS>(Kj, Bc * d_padded);
        zero_smem<T, THREADS>(Vj, Bc * d_padded);
        __syncthreads();
        if (d_aligned) {
            load_chunk_async_strided<T, THREADS>(Kj, K + kv_base + (long long)j0 * d, kv_rows, d, d_padded);
            load_chunk_async_strided<T, THREADS>(Vj, V + kv_base + (long long)j0 * d, kv_rows, d, d_padded);
            cp_async_commit(); cp_async_wait<0>();
        } else {
            load_strided_regular<T, THREADS>(Kj, K + kv_base + (long long)j0 * d, kv_rows, d, d_padded);
            load_strided_regular<T, THREADS>(Vj, V + kv_base + (long long)j0 * d, kv_rows, d, d_padded);
        }
        __syncthreads();

        if (!warp_active) { __syncthreads(); continue; }

        for (int tn = 0; tn < (Bc / 16); tn++) {
            const int tile_n = tn * 16;
            bool tile_skip = (j0 + tile_n >= N) || (causal && (i0 + warp_row_start + 15) < (j0 + tile_n));
            if (tile_skip) continue;

            // 1. Q x K^T → ScoreAcc in registers
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> ScoreAcc;
            wmma::fill_fragment(ScoreAcc, 0.f);
            #pragma unroll
            for (int t = 0; t < 128; t += 16) {
                if (t >= d) break;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, WMMA_T, wmma::col_major> fB;
                wmma::load_matrix_sync(fB, Kj + tile_n * d_padded + t, d_padded);
                wmma::mma_sync(ScoreAcc, fQ[t/16], fB, ScoreAcc);
            }

            // 2. Register-level softmax (no SMEM round-trip for scores)
            // Scale scores directly in registers
            float s0 = ScoreAcc.x[0] * scaleF;
            float s1 = ScoreAcc.x[1] * scaleF;
            float s2 = ScoreAcc.x[2] * scaleF;
            float s3 = ScoreAcc.x[3] * scaleF;
            float s4 = ScoreAcc.x[4] * scaleF;
            float s5 = ScoreAcc.x[5] * scaleF;
            float s6 = ScoreAcc.x[6] * scaleF;
            float s7 = ScoreAcc.x[7] * scaleF;

            // Causal + OOB masking in registers
            {
                const int abs_row_low  = i0 + warp_row_start + r_idx_low;
                const int abs_row_high = i0 + warp_row_start + r_idx_high;
                const int c_base = j0 + tile_n;
                const int c0 = c_base + (lane & 3) * 2;
                const int c1 = c0 + 1;
                const int c4 = c0 + 8;
                const int c5 = c4 + 1;

                if (abs_row_low >= N) {
                    s0 = -INFINITY; s1 = -INFINITY; s4 = -INFINITY; s5 = -INFINITY;
                } else {
                    if (c0 >= N || (causal && abs_row_low < c0)) s0 = -INFINITY;
                    if (c1 >= N || (causal && abs_row_low < c1)) s1 = -INFINITY;
                    if (c4 >= N || (causal && abs_row_low < c4)) s4 = -INFINITY;
                    if (c5 >= N || (causal && abs_row_low < c5)) s5 = -INFINITY;
                }
                if (abs_row_high >= N) {
                    s2 = -INFINITY; s3 = -INFINITY; s6 = -INFINITY; s7 = -INFINITY;
                } else {
                    if (c0 >= N || (causal && abs_row_high < c0)) s2 = -INFINITY;
                    if (c1 >= N || (causal && abs_row_high < c1)) s3 = -INFINITY;
                    if (c4 >= N || (causal && abs_row_high < c4)) s6 = -INFINITY;
                    if (c5 >= N || (causal && abs_row_high < c5)) s7 = -INFINITY;
                }
            }

            // Row max via warp shuffle (4 threads per row, lane%4)
            float tmax_low = fmaxf(fmaxf(s0, s1), fmaxf(s4, s5));
            tmax_low = fmaxf(tmax_low, __shfl_xor_sync(0xffffffff, tmax_low, 1));
            tmax_low = fmaxf(tmax_low, __shfl_xor_sync(0xffffffff, tmax_low, 2));

            float tmax_high = fmaxf(fmaxf(s2, s3), fmaxf(s6, s7));
            tmax_high = fmaxf(tmax_high, __shfl_xor_sync(0xffffffff, tmax_high, 1));
            tmax_high = fmaxf(tmax_high, __shfl_xor_sync(0xffffffff, tmax_high, 2));

            // Online softmax update
            float m_new_low = fmaxf(mi_low, tmax_low);
            float alpha_low = (mi_low == -INFINITY) ? 0.f : __expf(mi_low - m_new_low);
            if (mi_low == -INFINITY && m_new_low == -INFINITY) alpha_low = 1.0f;

            float m_new_high = fmaxf(mi_high, tmax_high);
            float alpha_high = (mi_high == -INFINITY) ? 0.f : __expf(mi_high - m_new_high);
            if (mi_high == -INFINITY && m_new_high == -INFINITY) alpha_high = 1.0f;

            // Compute P = exp(score - m_new) directly in registers
            float p0 = (s0 == -INFINITY) ? 0.f : __expf(s0 - m_new_low);
            float p1 = (s1 == -INFINITY) ? 0.f : __expf(s1 - m_new_low);
            float p4 = (s4 == -INFINITY) ? 0.f : __expf(s4 - m_new_low);
            float p5 = (s5 == -INFINITY) ? 0.f : __expf(s5 - m_new_low);

            float p2 = (s2 == -INFINITY) ? 0.f : __expf(s2 - m_new_high);
            float p3 = (s3 == -INFINITY) ? 0.f : __expf(s3 - m_new_high);
            float p6 = (s6 == -INFINITY) ? 0.f : __expf(s6 - m_new_high);
            float p7 = (s7 == -INFINITY) ? 0.f : __expf(s7 - m_new_high);

            // Row sum via warp shuffle
            float tsum_low = p0 + p1 + p4 + p5;
            tsum_low += __shfl_xor_sync(0xffffffff, tsum_low, 1);
            tsum_low += __shfl_xor_sync(0xffffffff, tsum_low, 2);

            float tsum_high = p2 + p3 + p6 + p7;
            tsum_high += __shfl_xor_sync(0xffffffff, tsum_high, 1);
            tsum_high += __shfl_xor_sync(0xffffffff, tsum_high, 2);

            // Update running softmax state
            li_low = li_low * alpha_low + tsum_low;
            mi_low = m_new_low;
            li_high = li_high * alpha_high + tsum_high;
            mi_high = m_new_high;

            // Rescale accumulators directly in registers (no SMEM broadcast)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k * 16 < d) {
                    acc[k].x[0] *= alpha_low;  acc[k].x[1] *= alpha_low;
                    acc[k].x[4] *= alpha_low;  acc[k].x[5] *= alpha_low;
                    acc[k].x[2] *= alpha_high; acc[k].x[3] *= alpha_high;
                    acc[k].x[6] *= alpha_high; acc[k].x[7] *= alpha_high;
                }
            }

            // Write P to SMEM for WMMA load (single write, no zero-init needed)
            S_P_warp[r_idx_low  * tile_stride + (lane & 3) * 2]     = T(p0);
            S_P_warp[r_idx_low  * tile_stride + (lane & 3) * 2 + 1] = T(p1);
            S_P_warp[r_idx_low  * tile_stride + (lane & 3) * 2 + 8] = T(p4);
            S_P_warp[r_idx_low  * tile_stride + (lane & 3) * 2 + 9] = T(p5);
            S_P_warp[r_idx_high * tile_stride + (lane & 3) * 2]     = T(p2);
            S_P_warp[r_idx_high * tile_stride + (lane & 3) * 2 + 1] = T(p3);
            S_P_warp[r_idx_high * tile_stride + (lane & 3) * 2 + 8] = T(p6);
            S_P_warp[r_idx_high * tile_stride + (lane & 3) * 2 + 9] = T(p7);
            __syncwarp();

            // 3. P x V
            wmma::fragment<wmma::matrix_a, 16, 16, 16, WMMA_T, wmma::row_major> fP;
            wmma::load_matrix_sync(fP, S_P_warp, tile_stride);
            #pragma unroll
            for (int t = 0; t < 128; t += 16) {
                if (t >= d) break;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, WMMA_T, wmma::row_major> fV;
                wmma::load_matrix_sync(fV, Vj + tile_n * d_padded + t, d_padded);
                wmma::mma_sync(acc[t/16], fP, fV, acc[t/16]);
            }
        }
        __syncthreads();
    }

    if (!warp_active) return;

    // Final store: directly from accumulator registers → global memory (no SMEM)
    // Use store_v2 only when d is even (guarantees __half2 alignment); scalar otherwise.
    float inv_li_low  = (li_low  > 0.f) ? 1.0f / li_low  : 0.f;
    float inv_li_high = (li_high > 0.f) ? 1.0f / li_high : 0.f;
    const bool d_even = (d % 2 == 0);

    #pragma unroll
    for (int t = 0; t < 128; t += 16) {
        if (t >= d) break;
        const int k = t / 16;
        const int c_lo = t + (lane & 3) * 2;
        const int c_hi = t + (lane & 3) * 2 + 8;

#define STORE_PAIR(row_o, col, v0, v1) do { \
    if ((col) + 1 < d) { \
        if (d_even) store_v2<T>((row_o) + (col), (v0), (v1)); \
        else { (row_o)[(col)] = T(v0); (row_o)[(col)+1] = T(v1); } \
    } else if ((col) < d) { \
        (row_o)[(col)] = T(v0); \
    } \
} while(0)

        if (i0 + warp_row_start + r_idx_low < N) {
            T* row_o = out_ptr + r_idx_low * d;
            STORE_PAIR(row_o, c_lo, acc[k].x[0] * inv_li_low, acc[k].x[1] * inv_li_low);
            STORE_PAIR(row_o, c_hi, acc[k].x[4] * inv_li_low, acc[k].x[5] * inv_li_low);
        }
        if (i0 + warp_row_start + r_idx_high < N) {
            T* row_o = out_ptr + r_idx_high * d;
            STORE_PAIR(row_o, c_lo, acc[k].x[2] * inv_li_high, acc[k].x[3] * inv_li_high);
            STORE_PAIR(row_o, c_hi, acc[k].x[6] * inv_li_high, acc[k].x[7] * inv_li_high);
        }
#undef STORE_PAIR
    }

    // Store L (one write per row via lane%4==0)
    if (lane % 4 == 0) {
        if (i0 + warp_row_start + r_idx_low < N)
            l_ptr[r_idx_low] = mi_low + logf(li_low);
        if (i0 + warp_row_start + r_idx_high < N)
            l_ptr[r_idx_high] = mi_high + logf(li_high);
    }
}

extern "C" void launch_flashattn_warp(const void* Q, const void* K, const void* V, void* O, 
    float* L, int N, int d, int batch, int heads, 
    bool causal, bool is_bf16, float dropout_p, cudaStream_t stream) {
    // 4 warps × 16 rows = Br=64, Bc=32
    int Br = 64, Bc = 32;
    if (d <= 32) { Br = 64; Bc = 64; }

    dim3 grid((N + Br - 1) / Br, heads, batch);
    dim3 block(THREADS); // THREADS = 128 (4 warps)

#define LNC(T_, WT_, BC_, BR_) do { \
    const int d_padded = max(((d + 15) / 16) * 16 + 8, 24); \
    const int tile_stride = 16 + 8; \
    size_t s = (size_t)(BR_ + 2 * BC_) * d_padded * sizeof(T_) + (size_t)WARPS_PER_BLOCK * tile_stride * 16 * sizeof(T_); \
    void(*f)(const T_*, const T_*, const T_*, T_*, float*, int, int, int, bool, float, uint32_t, uint32_t) = flashattn_wmma_kernel<T_, WT_, BC_, BR_>; \
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s); \
    f<<<grid, block, s, stream>>>((const T_*)Q, (const T_*)K, (const T_*)V, (T_*)O, L, N, d, heads, causal, dropout_p, 42, 43); \
} while(0)

    if (is_bf16) {
        LNC(__nv_bfloat16, __nv_bfloat16, 32, 64);
    } else {
        LNC(__half, __half, 32, 64);
    }
}
    }
}


