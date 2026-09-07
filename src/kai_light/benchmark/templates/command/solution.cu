#include <iostream>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
__global__ void transform(const int* input, int* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] * input[i] + 1;
}
#define CUDA_CHECK(call) do { if ((call) != cudaSuccess) return 2; } while (0)
#endif

int main() {
    int count;
    if (!(std::cin >> count) || count <= 0 || count > 10000000) return 1;
    std::vector<int> input(count), output(count);
    for (int& value : input) if (!(std::cin >> value)) return 1;
#ifdef __CUDACC__
    int *device_input = nullptr, *device_output = nullptr;
    size_t bytes = count * sizeof(int);
    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
    transform<<<(count + 255) / 256, 256>>>(device_input, device_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
#else
    for (int i = 0; i < count; ++i) output[i] = input[i] * input[i] + 1;
#endif
    std::cout << "{\"output\":[";
    for (int i = 0; i < count; ++i) {
        if (i) std::cout << ',';
        std::cout << output[i];
    }
    std::cout << "]}\n";
    return 0;
}
