// Kernel 1 of the epilogue: y[r, c] = x[r, c] + bias[c].
#include <cuda_runtime.h>

namespace {

__global__ void bias_add_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ y, int rows, int cols) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * cols;
    if (index < total) {
        y[index] = x[index] + bias[index % cols];
    }
}

}  // namespace

// Returns 0 on success. The caller reads launch errors through cudaGetLastError.
int launch_bias_add(const float* x, const float* bias, float* y, int rows, int cols,
                    cudaStream_t stream) {
    const long long total = static_cast<long long>(rows) * cols;
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((total + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, rows, cols);
    return 0;
}
