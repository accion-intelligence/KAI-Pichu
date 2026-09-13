// Kernel 3 of the epilogue: out = z + residual.
#include <cuda_runtime.h>

namespace {

__global__ void residual_add_kernel(const float* __restrict__ z, const float* __restrict__ residual,
                                    float* __restrict__ out, long long total) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total) {
        out[index] = z[index] + residual[index];
    }
}

}  // namespace

// Returns 0 on success. The caller reads launch errors through cudaGetLastError.
int launch_residual_add(const float* z, const float* residual, float* out, long long total,
                        cudaStream_t stream) {
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((total + threads - 1) / threads);
    residual_add_kernel<<<blocks, threads, 0, stream>>>(z, residual, out, total);
    return 0;
}
