/*
 * Scaled dot-product attention forward, FP16, head_dim = 64.
 *
 * Baseline: PyTorch's fused scaled_dot_product_attention through the ATen C++
 * API. For FP16 inputs PyTorch dispatches to its flash-attention, cuDNN or
 * CUTLASS memory-efficient kernel, all with FP32 accumulation. The tensors
 * wrap the caller's buffers; the only extra work is copying the result into o.
 */
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

// Returns 0 on success, -1 for unsupported arguments, -2 if PyTorch raised.
// Launch errors are read by the caller through cudaGetLastError.
int launch_attention_forward(
    const __half* q, const __half* k, const __half* v, __half* o,
    int batch, int heads, int seq, int head_dim, float scale, int causal,
    void* /*workspace*/, size_t /*workspace_bytes*/, cudaStream_t stream) {
    if (batch <= 0 || heads <= 0 || seq <= 0 || head_dim != 64) {
        return -1;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return -2;
    }
    try {
        const c10::cuda::CUDAStreamGuard guard(
            c10::cuda::getStreamFromExternal(stream, static_cast<c10::DeviceIndex>(device)));
        const auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, device);
        const std::vector<int64_t> shape{batch, heads, seq, head_dim};
        const at::Tensor query = at::from_blob(const_cast<__half*>(q), shape, options);
        const at::Tensor key = at::from_blob(const_cast<__half*>(k), shape, options);
        const at::Tensor value = at::from_blob(const_cast<__half*>(v), shape, options);
        at::Tensor out = at::from_blob(o, shape, options);
        out.copy_(at::scaled_dot_product_attention(
            query, key, value, /*attn_mask=*/{}, /*dropout_p=*/0.0, /*is_causal=*/causal != 0,
            /*scale=*/static_cast<double>(scale)));
    } catch (const std::exception&) {
        return -2;
    }
    return 0;
}
