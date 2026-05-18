#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>
#include <stdint.h>
#include "include/core/Tensor.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
    namespace cuda{
    template<typename T, int vec_size>
    struct alignas(sizeof(T) * vec_size) VectorType {
        T val[vec_size];
    };

    template<typename T>
    inline int get_alignment_size(const T* ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if(addr % 16 == 0) return 16;
        if(addr % 8 == 0) return 8;
        if(addr % 4 == 0) return 4;
        return 2;
    }

    template<int vec_size, typename Func, typename OutT, typename... InTs>
    __global__ void vectorized_kernel_impl(int N, Func f, OutT* out, const InTs*... ins) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;

        int vec_N = N/vec_size;

        for(int i=idx; i < vec_N; i+=stride) {
            int base_idx = i * vec_size;

            VectorType<OutT, vec_size> out_v;

            #pragma unroll
            for(int v=0; v < vec_size; ++v) {
                out_v.val[v] = f( (ins[base_idx + v])... );
            }

            reinterpret_cast<VectorType<OutT, vec_size>*>(out)[i] = out_v;
        }

        int tail_start = vec_N * vec_size;
        for(int i= tail_start + idx; i < N; i+=stride) {
            out[i] = f( (ins[i])... );
        }
    }


  /**
 * @brief Internal launch function using raw pointers
 */
template<typename Func, typename OutT, typename... InTs>
void launch_impl(int N, Func f, OutT* out, const InTs*... ins) {
    if (N <= 0) return;

    // 1. Determine Vector Size based on input alignments and derived output type
    int min_align = get_alignment_size(out);
    int in_aligns[] = { get_alignment_size(ins)... };
    for (int a : in_aligns) if (a < min_align) min_align = a;

    // Use sizeof(OutT) for vector size calculation
    int vec_size = min_align / sizeof(OutT);

    if (vec_size >= 16) vec_size = 16;
    else if (vec_size >= 8) vec_size = 8;
    else if (vec_size >= 4) vec_size = 4;
    else if (vec_size >= 2) vec_size = 2;
    else vec_size = 1;

    // 2. Use CUDA Occupancy API to find optimal block size
    int blockSize;   // Optimal threads per block
    int minGridSize; // Min blocks to saturate all SMs

    // We must handle the dispatch here to pass the correct kernel template to the API
    auto kernel_ptr = (vec_size == 16) ? vectorized_kernel_impl<16, Func, OutT, InTs...> :
                      (vec_size == 8) ? vectorized_kernel_impl<8, Func, OutT, InTs...> :
                      (vec_size == 4) ? vectorized_kernel_impl<4, Func, OutT, InTs...> :
                      (vec_size == 2) ? vectorized_kernel_impl<2, Func, OutT, InTs...> :
                                        vectorized_kernel_impl<1, Func, OutT, InTs...>;

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, 
        &blockSize, 
        (void*)kernel_ptr, 
        0, // Dynamic shared memory
        0  // Block size limit (0 = let CUDA decide)
    );

    // 3. Calculate Grid Size (Saturate the GPU)
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    int blocks = std::min(static_cast<int64_t>((N / vec_size + blockSize - 1) / blockSize), 
                      static_cast<int64_t>(numSMs * 32));

    if (blocks == 0) blocks = 1;

    // 4. Dispatch with Optimal Parameters
    if (vec_size == 16) {
        vectorized_kernel_impl<16><<<blocks, blockSize>>>(N, f, out, ins...);
    } else if (vec_size == 8) {
        vectorized_kernel_impl<8><<<blocks, blockSize>>>(N, f, out, ins...);
    } else if (vec_size == 4) {
        vectorized_kernel_impl<4><<<blocks, blockSize>>>(N, f, out, ins...);
    } else if (vec_size == 2) {
        vectorized_kernel_impl<2><<<blocks, blockSize>>>(N, f, out, ins...);
    } else {
        vectorized_kernel_impl<1><<<blocks, blockSize>>>(N, f, out, ins...);
    }
}

/**
 * @brief Launch the universal vectorized kernel using Tensor objects
 */
template<typename Func, typename... Tensors>
Tensor& launch(int N, Func f, Tensor& out, const Tensors&... ins) {
    Dtype in_dtype = out.dtype();

    if (in_dtype == Dtype::Float32) {
        launch_impl<Func, float, float>(N, f, out.data<float>(), ins.template data<float>()...);
    } else if (in_dtype == Dtype::Float16) {
        launch_impl<Func, __half, __half>(N, f, out.data<__half>(), ins.template data<__half>()...);
    } else if (in_dtype == Dtype::Bfloat16) {
        launch_impl<Func, __nv_bfloat16, __nv_bfloat16>(N, f, out.data<__nv_bfloat16>(), ins.template data<__nv_bfloat16>()...);
    }
    return out;
}

} // namespace cuda
} // namespace OwnTensor

