# CUDA Kernels Collection

A collection of high-performance CUDA kernels focused on deep learning primitives and transformer optimization techniques.

This repository contains optimized implementations of normalization, activation, vectorization utilities, and FlashAttention-style kernels commonly used in modern LLM architectures.

---

# Files Overview

## FlashAttention.cu

Implementation of FlashAttention-style attention kernels optimized for GPU memory efficiency and throughput.

### Features
- Tiled attention computation
- Shared memory optimization
- Online softmax computation
- Reduced HBM memory access
- Improved arithmetic intensity
- Warp-level parallelism
- Numerically stable softmax accumulation

### Purpose
Traditional attention mechanisms materialize the full attention matrix in global memory, which becomes bandwidth-bound and memory expensive for large sequence lengths.

This kernel follows the FlashAttention approach:
- Computes attention block-by-block
- Avoids storing full attention scores
- Performs fused softmax and accumulation
- Minimizes global memory traffic

### Optimization Techniques
- Shared memory tiling
- Register blocking
- Vectorized memory loads
- Warp reductions
- Memory coalescing
- Online normalization

---

## LayerNormKernels.cu

CUDA implementation of Layer Normalization kernels used in transformer architectures.

### Features
- Mean and variance computation
- Parallel reduction
- Warp-level synchronization
- Vectorized reads/writes
- Fused normalization operations

### Purpose
LayerNorm normalizes activations across feature dimensions to stabilize training and improve convergence.

The kernel efficiently computes:
- Mean
- Variance
- Standard deviation
- Affine transformation

using GPU parallelism.

### Optimization Techniques
- Shared memory reductions
- Warp shuffle reductions
- Coalesced memory access
- Float4 vectorization
- Fused operations to reduce kernel launches

---

## Rmsnorm.cu

CUDA implementation of RMSNorm (Root Mean Square Normalization).

### Features
- RMS-based normalization
- Reduced computational overhead
- Transformer inference optimization
- Vectorized memory operations

### Purpose
RMSNorm is a simplified normalization technique used in many modern LLMs such as:
- LLaMA
- Mistral
- Gemma

Unlike LayerNorm, RMSNorm removes mean subtraction and only scales using RMS statistics.

### Advantages
- Lower computational cost
- Better inference efficiency
- Reduced synchronization overhead
- Improved throughput on GPUs

### Optimization Techniques
- Warp-level reductions
- Shared memory accumulation
- Float4 vectorized loads
- Register-level accumulation

---

## Swiglu.cu

CUDA implementation of SwiGLU activation used in transformer feed-forward networks.

### Features
- SwiGLU fused activation
- Sigmoid-weighted gating
- Vectorized computation
- Transformer FFN optimization

### Purpose
SwiGLU is a gated activation function commonly used in modern LLM architectures.

It combines:
- Swish activation
- Gated linear units

to improve expressiveness and training performance.

### Formula

```math
SwiGLU(x, g) = Swish(g) \cdot x
```

where

```math
Swish(x) = x \cdot \sigma(x)
```

### Optimization Techniques
- Fused activation computation
- Vectorized loads/stores
- Reduced kernel launch overhead
- Memory coalescing

---

## vectorization.cuh

Header utilities for vectorized CUDA operations.

### Features
- `float4` vectorized memory access
- Alignment utilities
- Packed data loading/storing
- SIMD-style GPU operations

### Purpose
Vectorization improves GPU throughput by processing multiple elements per instruction.

Instead of:
```cpp
float x = input[i];
```

vectorized kernels use:
```cpp
float4 x = reinterpret_cast<float4*>(input)[i];
```

This:
- Reduces memory transactions
- Improves bandwidth utilization
- Increases arithmetic throughput
- Enhances coalesced access patterns

### Common Usage
- LayerNorm
- RMSNorm
- Activation kernels
- Matrix operations
- Attention kernels

---

# Key CUDA Concepts Used

- Shared Memory
- Warp-Level Parallelism
- Warp Shuffle Instructions
- Thread Block Tiling
- Register Blocking
- Vectorized Memory Access
- Memory Coalescing
- Online Softmax
- Parallel Reductions
- Kernel Fusion

---

# Target Applications

These kernels are useful for:
- Transformer models
- GPT-style architectures
- LLM inference engines
- CUDA performance learning
- GPU systems programming
- Deep learning optimization research

---

# Build Requirements

- CUDA Toolkit
- NVIDIA GPU with CUDA support
- C++17 compatible compiler
- Linux or Windows with NVCC

---

# Learning Goals

This repository demonstrates:
- Writing custom CUDA kernels
- GPU memory optimization
- Transformer kernel engineering
- Efficient reduction algorithms
- Vectorized GPU programming
- High-performance deep learning primitives
