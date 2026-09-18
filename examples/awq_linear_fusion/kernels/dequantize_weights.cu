/*
 * dequantize_weights: AWQ 4-bit weight dequantization to FP16.
 *
 * Copied verbatim from casper-hansen/AutoAWQ_kernels
 * (awq_ext/quantization/gemm_cuda_gen.cu), MIT License, Copyright (c) 2023
 * Casper; see licenses/AutoAWQ_kernels-MIT.txt. The kernel itself credits
 * compressa-ai/AutoAWQ, and dequantize.cuh is modified from NVIDIA
 * FasterTransformer. Only the launcher after the kernel is KAI code: it
 * reproduces the default launch configuration of AutoAWQ's host wrapper
 * (8x8 threads, one thread per packed int32) without the PyTorch tensor API.
 *
 * Layout (AWQ): qweight int32 [K, N/8]; qzeros int32 [K/G, N/8]; scales fp16
 * [K/G, N]; output fp16 [K, N]. Each int32 packs eight 4-bit values whose
 * column order is 0, 4, 1, 5, 2, 6, 3, 7 by nibble position.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "dequantize.cuh"

// Dequantization to fp16
// kernel
// Source - https://github.com/compressa-ai/AutoAWQ/blob/6673333456b8871522b11a7fb110de612edfdf95/awq_cuda/quantization/gemm_cuda_gen.cu#L32C1-L32C1
__global__ void __launch_bounds__(64) dequantize_weights(int* __restrict__ B, // 4096x64    4096 rows    64 cols
                                                         half* __restrict__ scaling_factors,  // 32x512   32 rows    512 cols
                                                         int* __restrict__ zeros,  // 32x64    32 rows     64 cols
                                                         half* __restrict__ C, // 4096x512    4096 rows    512 cols
                                                         int G,
                                                         int in_c,
                                                         int out_c)
{
  if (blockIdx.z > 0) {
    B = B + blockIdx.z * in_c * out_c / 8;
    scaling_factors = scaling_factors + blockIdx.z * in_c * out_c / G;
    zeros = zeros + blockIdx.z * in_c * out_c / G / 8;
    C = C + blockIdx.z * in_c * out_c;
  }
  int j_factors1 = 4;
  int row_stride2 = 4;
  int split_k_iters = 1;
  static constexpr uint32_t ZERO = 0x0;
  half B_shared[32 * (128 + 8)];

  half* B_shared_ptr2 = B_shared;

  half B_shared_warp[32];
  int OC = 512;

  int N = blockDim.x * gridDim.x;  // 2
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;  // + i (<8)
  half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;  // + i (<8)
  half* scaling_factors_ptr2 = scaling_factors + index4;


    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);
int j=0;

      uint32_t B_loaded = *(uint32_t*)(B_ptr2 + j);
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

      *(uint4*)(B_shared_ptr2 + j) = B_loaded_fp16;

  for (int i=0; i<8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

// KAI launcher. Returns 0 on success, -1 for shapes the kernel's default
// launch configuration does not cover. Launch errors are read by the caller.
int launch_dequantize_weights(const int* qweight, const __half* scales, const int* qzeros,
                              __half* out, int k, int n, int group_size, cudaStream_t stream) {
    const int packed_columns = n / 8;
    if (k <= 0 || n <= 0 || n % 64 != 0 || k % 8 != 0 || group_size <= 0 || k % group_size != 0) {
        return -1;
    }
    const dim3 threads(8, 8);
    const dim3 blocks(packed_columns / 8, k / 8, 1);
    dequantize_weights<<<blocks, threads, 0, stream>>>(
        const_cast<int*>(qweight), const_cast<half*>(scales), const_cast<int*>(qzeros), out,
        group_size, k, n);
    return 0;
}
