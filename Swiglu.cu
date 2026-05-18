#include <iostream>
#include <cuda_runtime.h>

__device__ float sigmoid(const float X) {
    return 1.f/(1 + expf(-X));
}

__device__ float swish(const float X) {
    return X * sigmoid(X);
}

__global__ void __launch_bounds__(256, 8)
swiglu(const float4* __restrict__ X, float4* __restrict__ Y, size_t m, size_t n, int cols4) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= m || col >= cols4) return;

    int row_in4  = row * (2 * cols4);
    int row_out4 = row * cols4;

    float4 a = __ldg(&X[row_in4 + col]);
    float4 b = __ldg(&X[row_in4 + cols4 + col]);

    float4 y;
    y.x = swish(a.x) * b.x;
    y.y = swish(a.y) * b.y;
    y.z = swish(a.z) * b.z;
    y.w = swish(a.w) * b.w;

    Y[row_out4 + col] = y;
}

__global__ void swiglu_tail(const float* X, float* Y, size_t m, size_t n, int col_start) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = col_start + blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= m || col >= n) return;

    float a = X[row * 2 * n + col];
    float b = X[row * 2 * n + n + col];

    Y[row * n + col] = swish(a) * b;
}

extern "C" void launcher(const float* X, float* Y, size_t m, size_t n) {

    int cols4     = n / 4;
    int cols_rem  = n & 3;
    int col_start = cols4 * 4;

    dim3 block(32, 4);

    if(cols4 > 0) {
        dim3 grid((cols4 + block.x - 1) / block.x,
                  (m     + block.y - 1) / block.y);

        swiglu<<<grid, block>>>(reinterpret_cast<const float4*>(X),
                                reinterpret_cast<float4*>(Y),
                                m, n, cols4);
    }

    if(cols_rem > 0) {
        dim3 grid((cols_rem + block.x - 1) / block.x,
                  (m        + block.y - 1) / block.y);

        swiglu_tail<<<grid, block>>>(X, Y, m, n, col_start);
    }
}
