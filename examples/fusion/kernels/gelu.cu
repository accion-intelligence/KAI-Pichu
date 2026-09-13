// Kernel 2 of the epilogue: z = gelu(y), exact erf form: 0.5 * y * (1 + erf(y / sqrt(2))).
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_erf(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

__global__ void gelu_kernel(const float* __restrict__ y, float* __restrict__ z, long long total) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total) {
        z[index] = gelu_erf(y[index]);
    }
}

}  // namespace

// Returns 0 on success. The caller reads launch errors through cudaGetLastError.
int launch_gelu(const float* y, float* z, long long total, cudaStream_t stream) {
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((total + threads - 1) / threads);
    gelu_kernel<<<blocks, threads, 0, stream>>>(y, z, total);
    return 0;
}
