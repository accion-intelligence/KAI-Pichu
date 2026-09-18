// Stage 2 of the unfused pipeline: y[M, N] = x[M, K] · W[K, N] in FP16 with
// FP32 accumulation through cuBLAS. This is the library call AutoAWQ makes
// (torch.matmul) after dequantizing the weights, expressed without PyTorch.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

cublasHandle_t cublas_handle() {
    static cublasHandle_t shared = nullptr;
    if (shared == nullptr) {
        cublasCreate(&shared);
    }
    return shared;
}

}  // namespace

// Row-major C = A · B is computed as column-major Cᵀ = Bᵀ · Aᵀ, so cuBLAS sees
// W as the [N, K] left operand and x as the [K, M] right operand.
// Returns 0 on success, -2 for a cuBLAS error.
int launch_fp16_gemm(const __half* x, const __half* w, __half* y, int m, int k, int n,
                     cudaStream_t stream) {
    cublasHandle_t handle = cublas_handle();
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return -2;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, w, CUDA_R_16F, n, x, CUDA_R_16F, k,
        &beta, y, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -2;
}
