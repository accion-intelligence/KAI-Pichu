/*
 * Unfused baseline: AutoAWQ's dequantize_weights kernel materializes the FP16
 * weight matrix in the workspace, then cuBLAS multiplies. This is AutoAWQ's
 * own large-batch path (dequantize_weights_cuda followed by torch.matmul).
 * A fused implementation reads the packed INT4 weights, zeros and scales
 * directly inside the matrix-vector product and never writes W.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/dequantize_weights.cu"
#include "kernels/cublas_gemm.cu"

// Returns 0 on success, -1 if the workspace cannot hold the FP16 weights or a
// shape is unsupported, -2 for a cuBLAS error. Launch errors are read by the
// caller through cudaGetLastError.
int launch_awq_linear(const __half* x, const int* qweight, const int* qzeros, const __half* scales,
                      __half* y, int m, int k, int n, int group_size,
                      void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    const size_t weight_bytes = static_cast<size_t>(k) * n * sizeof(__half);
    if (m <= 0 || workspace_bytes < weight_bytes) {
        return -1;
    }
    __half* dequantized = static_cast<__half*>(workspace);
    int status = launch_dequantize_weights(qweight, scales, qzeros, dequantized, k, n, group_size, stream);
    if (status == 0) {
        status = launch_fp16_gemm(x, dequantized, y, m, k, n, stream);
    }
    return status;
}
