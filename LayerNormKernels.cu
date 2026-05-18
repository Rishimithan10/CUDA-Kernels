#include "ops/helpers/LayerNormKernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Unused headers removed

namespace OwnTensor {
namespace cuda {

// =================================================================================
// Helper: Warp Reduction
// =================================================================================
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// =================================================================================
// Welford Helpers
// =================================================================================
template<typename AccT>
struct WelfordData {
    AccT n;
    AccT mu;
    AccT m2;
};

template<typename AccT>
__device__ __inline__ WelfordData<AccT> welford_merge(WelfordData<AccT> a, WelfordData<AccT> b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData<AccT> res;
    res.n  = a.n + b.n;
    AccT delta = b.mu - a.mu;
    res.mu = a.mu + delta * (b.n / res.n);
    res.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / res.n);
    return res;
}

// =================================================================================
// Forward Kernel  —  Welford + Vectorized Loads (float4 / half2×4 / bfloat162×4)
// =================================================================================
// Grid: [rows], Block: [256] (8 warps)
// One block per row; each thread strides over columns.
template<typename T, typename AccT>
__global__ void layer_norm_forward_kernel(
    const T*    __restrict__ x,
    const T*    __restrict__ gamma,
    const T*    __restrict__ beta,
    T*          __restrict__ y,
    AccT*       __restrict__ mean_out,
    AccT*       __restrict__ rstd_out,
    int cols,
    AccT eps)
{
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const T* row_x   = x + row * cols;
    T*       row_y   = y + row * cols;

    // --- PHASE 1: VECTORIZED LOCAL ACCUMULATION ---
    AccT local_sum    = 0;
    AccT local_sq_sum = 0;
    int  local_count  = 0;

    if constexpr (std::is_same_v<T, float>) {
        // float4: 128-bit load = 4 × fp32
        const float4* x_vec  = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 v = x_vec[i];
            AccT vx = v.x, vy = v.y, vz = v.z, vw = v.w;
            local_sum    += vx + vy + vz + vw;
            local_sq_sum += vx*vx + vy*vy + vz*vz + vw*vw;
            local_count  += 4;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT val = row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }

    } else if constexpr (std::is_same_v<T, __half>) {
        // float4 reinterpreted as 4 × __half2: 128-bit load = 8 × fp16
        const float4* x_vec   = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(h[k]);
                local_sum    += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x * (AccT)f.x + (AccT)f.y * (AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // float4 reinterpreted as 4 × __nv_bfloat162: 128-bit load = 8 × bf16
        const float4* x_vec   = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(h[k]);
                local_sum    += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x * (AccT)f.x + (AccT)f.y * (AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    }

    // --- PHASE 2: CONVERT LOCAL SUMS TO WELFORD STATE ---
    // Avoids per-element division; numerically equivalent for typical LLM feature sizes.
    AccT n = (AccT)local_count;
    WelfordData<AccT> state = {
        n,
        (n > 0) ? local_sum / n : (AccT)0,
        (n > 0) ? (local_sq_sum - local_sum * local_sum / n) : (AccT)0
    };

    // --- PHASE 3: WARP-LEVEL REDUCTION (shuffle) ---
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        WelfordData<AccT> other;
        other.n  = __shfl_down_sync(0xffffffff, state.n,  offset);
        other.mu = __shfl_down_sync(0xffffffff, state.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        state = welford_merge(state, other);
    }

    // --- PHASE 4: BLOCK-LEVEL REDUCTION (shared memory) ---
    // Lane 0 of each warp writes its partial result; thread 0 merges all warps.
    __shared__ WelfordData<AccT> s_welford[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) s_welford[warp_id] = state;
    __syncthreads();

    if (tid == 0) {
        WelfordData<AccT> final_state = s_welford[0];
        const int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; ++i)
            final_state = welford_merge(final_state, s_welford[i]);
        s_welford[0] = final_state;
    }
    __syncthreads();

    // --- PHASE 5: FINAL STATISTICS ---
    const AccT mu   = s_welford[0].mu;
    const AccT rstd = rsqrtf(s_welford[0].m2 / cols + eps);

    if (tid == 0) {
        if (mean_out) mean_out[row] = mu;
        if (rstd_out) rstd_out[row] = rstd;
    }

    // --- PHASE 6: VECTORIZED NORMALIZE & WRITEOUT ---
    if constexpr (std::is_same_v<T, float>) {
        float4*       y_vec   = reinterpret_cast<float4*>(row_y);
        const float4* x_vec   = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xv = x_vec[i];
            float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1,1,1,1);
            float4 bv = beta  ? reinterpret_cast<const float4*>(beta)[i]  : make_float4(0,0,0,0);
            float4 res;
            res.x = ((AccT)xv.x - mu) * rstd * gv.x + bv.x;
            res.y = ((AccT)xv.y - mu) * rstd * gv.y + bv.y;
            res.z = ((AccT)xv.z - mu) * rstd * gv.z + bv.z;
            res.w = ((AccT)xv.w - mu) * rstd * gv.w + bv.w;
            y_vec[i] = res;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
        }

    } else if constexpr (std::is_same_v<T, __half>) {
        float4*       y_vec   = reinterpret_cast<float4*>(row_y);
        const float4* x_vec   = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __half2* xh = reinterpret_cast<const __half2*>(&xraw);

            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __half2* gh = gamma ? reinterpret_cast<const __half2*>(&graw) : nullptr;
            const __half2* bh = beta  ? reinterpret_cast<const __half2*>(&braw) : nullptr;

            float4  yraw;
            __half2* yh = reinterpret_cast<__half2*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __half22float2(xh[k]);
                xf.x = (xf.x - mu) * rstd;
                xf.y = (xf.y - mu) * rstd;
                if (gh) { float2 gf = __half22float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __half22float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22half2_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
        }

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        float4*       y_vec   = reinterpret_cast<float4*>(row_y);
        const float4* x_vec   = reinterpret_cast<const float4*>(row_x);
        const int     vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xraw);

            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __nv_bfloat162* gh = gamma ? reinterpret_cast<const __nv_bfloat162*>(&graw) : nullptr;
            const __nv_bfloat162* bh = beta  ? reinterpret_cast<const __nv_bfloat162*>(&braw) : nullptr;

            float4        yraw;
            __nv_bfloat162* yh = reinterpret_cast<__nv_bfloat162*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xh[k]);
                xf.x = (xf.x - mu) * rstd;
                xf.y = (xf.y - mu) * rstd;
                if (gh) { float2 gf = __bfloat1622float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __bfloat1622float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22bfloat162_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i] - mu) * rstd * g + b);
        }
    }
}


void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<float, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const __half* x,
    const __half* gamma,
    const __half* beta,
    __half* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<__half, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

void layer_norm_forward_cuda(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    __nv_bfloat16* y,
    float* mean,
    float* rstd,
    int rows,
    int cols,
    float eps)
{
    int threads = 256;
    layer_norm_forward_kernel<__nv_bfloat16, float><<<rows, threads>>>(x, gamma, beta, y, mean, rstd, cols, eps);
}

// =================================================================================
// Backward Kernels
// =================================================================================

// Kernel 1: Compute gradients for Gamma and Beta (Reduce over Rows)
// Grid: [cols], Block: [256]
// Each block handles one column (feature), reduces over all rows.
// Very simple implementation, might be slow for massive batch sizes but fine for GPT-2.
// __global__ void ln_backward_gamma_beta_kernel(
//     const float* __restrict__ grad_y,
//     const float* __restrict__ x,
//     const float* __restrict__ mean,
//     const float* __restrict__ rstd,
//     float* __restrict__ grad_gamma,
//     float* __restrict__ grad_beta,
//     int rows,
//     int cols)
// {
//     // Block handles 32 columns, 8 rows of threads (Total 256 threads)
//     int tx = threadIdx.x; // Column index within tile (0-31)
//     int ty = threadIdx.y; // Row index within tile (0-7)
    
//     // Shared memory for reduction across the 8 rows in the block
//     __shared__ float s_dgamma[8][32];
//     __shared__ float s_dbeta[8][32];
//     for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
//         int col = col_base + tx;
        
//         float d_gamma_acc = 0.0f;
//         float d_beta_acc = 0.0f;
//         // Cooperative row processing
//         if (col < cols) {
//             for (int row = ty; row < rows; row += 8) { // 8 is blockDim.y
//                 float gy = grad_y[row * cols + col]; // COALESCED!
//                 float input_val = x[row * cols + col]; // COALESCED!
//                 float m = mean[row];
//                 float rs = rstd[row];
                
//                 float norm_x = (input_val - m) * rs;
//                 d_beta_acc += gy;
//                 d_gamma_acc += gy * norm_x;
//             }
//         }
//         // Store partial sums in shared memory
//         s_dgamma[ty][tx] = d_gamma_acc;
//         s_dbeta[ty][tx] = d_beta_acc;
//         __syncthreads();
//         // Reduce across the 8 threads that handled different rows for this column
//         if (ty == 0 && col < cols) {
//             float final_dgamma = 0, final_dbeta = 0;
//             #pragma unroll
//             for (int i = 0; i < 8; i++) {
//                 final_dgamma += s_dgamma[i][tx];
//                 final_dbeta += s_dbeta[i][tx];
//             }
            
//             // Atomic add to global memory (each block handles a different set of rows/cols)
//             // If grid_stride over rows was applied, we'd need atomicAdd.
//             // Since we iterate over ALL rows in this loop, we can just write if we use block stride over cols.
//             // However, to be safe and support multiple blocks per col range:
//             atomicAdd(&grad_gamma[col], final_dgamma);
//             atomicAdd(&grad_beta[col], final_dbeta);
//         }
//         __syncthreads();
//     }
// }
__global__ void ln_backward_gamma_beta_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int rows,
    int cols)
{
    int tx = threadIdx.x; // Column offset (0-31)
    int ty = threadIdx.y; // Row offset (0-7)
    
    __shared__ float s_dgamma[8][32];
    __shared__ float s_dbeta[8][32];

    // 2D Grid Stride Loop: 
    // blockIdx.x handles columns, blockIdx.y handles rows
    #pragma unroll 4
    for (int col_base = blockIdx.x * 32; col_base < cols; col_base += gridDim.x * 32) {
        int col = col_base + tx;
        
        float d_gamma_acc = 0.0f;
        float d_beta_acc = 0.0f;

        if (col < cols) {
            // STRIDE OVER ROWS using gridDim.y
            // Each block in the Y-dimension handles a subset of rows
            for (int row = blockIdx.y * 8 + ty; row < rows; row += gridDim.y * 8) {
                float gy = grad_y[row * cols + col];
                float input_val = x[row * cols + col];
                float m = mean[row];
                float rs = rstd[row];
                
                float norm_x = (input_val - m) * rs;
                d_beta_acc += gy;
                d_gamma_acc += gy * norm_x;
            }
        }

        s_dgamma[ty][tx] = d_gamma_acc;
        s_dbeta[ty][tx] = d_beta_acc;
        __syncthreads();

        if (ty == 0 && col < cols) {
            float final_dgamma = 0, final_dbeta = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                final_dgamma += s_dgamma[i][tx];
                final_dbeta += s_dbeta[i][tx];
            }
            // MUST use atomicAdd as multiple blocks (gridDim.y) 
            // are now contributing to the same grad_gamma[col]
            atomicAdd(&grad_gamma[col], final_dgamma);
            atomicAdd(&grad_beta[col], final_dbeta);
        }
        __syncthreads();
    }
}

// Kernel 2: Compute Gradients for Input (Per Row)
// Standard derivation for LayerNorm backward
__global__ void ln_backward_input_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ grad_x,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* dy_row = grad_y + row * cols;
    const float* x_row = x + row * cols;
    float* dx_row = grad_x + row * cols;
    
    float m = mean[row];
    float rs = rstd[row];
    
    // 1. Compute local generic reductions: sum(dy * gamma) and sum(dy * gamma * (x-m))
    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_norm = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        sum_dy_gamma += dy * g;
        sum_dy_gamma_norm += dy * g * norm_x;
    }
    
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dy_gamma_norm = warpReduceSum(sum_dy_gamma_norm);
    
    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = 0; s_sum2 = 0; }
    __syncthreads();
    
    if (tid % warpSize == 0) {
        atomicAdd(&s_sum1, sum_dy_gamma);
        atomicAdd(&s_sum2, sum_dy_gamma_norm);
    }
    __syncthreads();
    
    float total_sum1 = s_sum1;
    float total_sum2 = s_sum2;
    
    // 2. Compute dx
    // dxhat = (dy * gamma)
    // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    //    = rstd * (dy*gamma - (1/D)*sum(dy*gamma) - xhat * (1/D)*sum(dy*gamma*xhat))
    float inv_cols = 1.0f / cols;
    
    #pragma unroll 4
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = (gamma) ? gamma[i] : 1.0f;
        float dy = dy_row[i];
        float val = x_row[i];
        float norm_x = (val - m) * rs;
        
        float term1 = dy * g;
        float term2 = total_sum1; 
        float term3 = norm_x * total_sum2;
        
        dx_row[i] = rs * (term1 - (term2 + term3) * inv_cols);
    }
}


void layer_norm_backward_cuda(
    const float* grad_y,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int cols)
{
    // 1. Gradients for Weights (Gamma/Beta)
    if (grad_gamma != nullptr || grad_beta != nullptr) {
        cudaMemset(grad_gamma, 0, cols * sizeof(float));
        cudaMemset(grad_beta, 0, cols * sizeof(float));

        dim3 threads(32, 8); // 256 threads per block
        
        // X handles Columns
        int blocks_x = (cols + 31) / 32;
        
        // Y handles Rows - aim for ~160-320 total blocks to saturate SMs
        // If blocks_x is 24, we can set blocks_y to 10 or 12.
        int blocks_y = 128 / blocks_x; 
        if (blocks_y < 1) blocks_y = 1;
        if (blocks_y > 32) blocks_y = 32; // Limit to avoid too much atomic contention

        dim3 grid(blocks_x, blocks_y);

        ln_backward_gamma_beta_kernel<<<grid, threads>>>(
            grad_y, x, mean, rstd, grad_gamma, grad_beta, rows, cols
        );
    }
    
    // 2. Gradients for Input
    if (grad_x != nullptr) {
        int threads = 256;
        if (cols > 256) threads = 512;
        ln_backward_input_kernel<<<rows, threads>>>(
            grad_y, x, mean, rstd, gamma, grad_x, cols
        );
    }
}

} // namespace cuda
} // namespace OwnTensor
