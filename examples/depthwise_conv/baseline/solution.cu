/*
 * Depthwise 7x7 convolution, NCHW FP16 with FP32 accumulation.
 *
 * Baseline: cuDNN's grouped convolution with group count = channels. The first
 * call for a shape runs cudnnFindConvolutionForwardAlgorithmEx to pick cuDNN's
 * fastest algorithm that fits the workspace and caches the choice; that
 * happens during the adapter's eager warm-up, before graph capture.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <map>
#include <tuple>

namespace {

constexpr int kKernel = 7;
constexpr int kPad = kKernel / 2;

struct Plan {
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_bytes;
};

using Shape = std::tuple<int, int, int, int>;

cudnnHandle_t handle() {
    static cudnnHandle_t shared = nullptr;
    if (shared == nullptr) {
        cudnnCreate(&shared);
    }
    return shared;
}

std::map<Shape, Plan>& plans() {
    static std::map<Shape, Plan> cache;
    return cache;
}

struct Descriptors {
    cudnnTensorDescriptor_t x = nullptr;
    cudnnTensorDescriptor_t y = nullptr;
    cudnnFilterDescriptor_t w = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;

    cudnnStatus_t create(int n, int c, int h, int width) {
        cudnnStatus_t status = cudnnCreateTensorDescriptor(&x);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateTensorDescriptor(&y);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateFilterDescriptor(&w);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateConvolutionDescriptor(&conv);
        if (status == CUDNN_STATUS_SUCCESS) {
            status = cudnnSetTensor4dDescriptor(x, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, width);
        }
        if (status == CUDNN_STATUS_SUCCESS) {
            status = cudnnSetTensor4dDescriptor(y, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, width);
        }
        if (status == CUDNN_STATUS_SUCCESS) {
            status = cudnnSetFilter4dDescriptor(w, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, c, 1, kKernel, kKernel);
        }
        if (status == CUDNN_STATUS_SUCCESS) {
            status = cudnnSetConvolution2dDescriptor(conv, kPad, kPad, 1, 1, 1, 1,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        }
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnSetConvolutionGroupCount(conv, c);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnSetConvolutionMathType(conv, CUDNN_DEFAULT_MATH);
        return status;
    }

    ~Descriptors() {
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
        if (w) cudnnDestroyFilterDescriptor(w);
        if (y) cudnnDestroyTensorDescriptor(y);
        if (x) cudnnDestroyTensorDescriptor(x);
    }
};

cudnnStatus_t choose_plan(const Descriptors& d, const __half* x, const __half* w, __half* y,
                          void* workspace, size_t workspace_bytes, Plan* plan) {
    constexpr int kRequested = 8;
    cudnnConvolutionFwdAlgoPerf_t results[kRequested];
    int returned = 0;
    const cudnnStatus_t status = cudnnFindConvolutionForwardAlgorithmEx(
        handle(), d.x, x, d.w, w, d.conv, d.y, y, kRequested, &returned, results,
        workspace, workspace_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
        return status;
    }
    for (int i = 0; i < returned; ++i) {
        if (results[i].status == CUDNN_STATUS_SUCCESS && results[i].memory <= workspace_bytes) {
            *plan = {results[i].algo, results[i].memory};
            return CUDNN_STATUS_SUCCESS;
        }
    }
    return CUDNN_STATUS_NOT_SUPPORTED;
}

}  // namespace

// Returns 0 on success, -1 for unsupported arguments, -2 for a cuDNN error.
// Launch errors are read by the caller through cudaGetLastError.
int launch_depthwise_conv(const __half* x, const __half* w, __half* y,
                          int n, int c, int h, int width,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (n <= 0 || c <= 0 || h <= 0 || width <= 0) {
        return -1;
    }
    if (cudnnSetStream(handle(), stream) != CUDNN_STATUS_SUCCESS) {
        return -2;
    }
    Descriptors d;
    if (d.create(n, c, h, width) != CUDNN_STATUS_SUCCESS) {
        return -2;
    }
    const Shape shape{n, c, h, width};
    auto& cache = plans();
    auto found = cache.find(shape);
    if (found == cache.end()) {
        Plan plan{};
        if (choose_plan(d, x, w, y, workspace, workspace_bytes, &plan) != CUDNN_STATUS_SUCCESS) {
            return -2;
        }
        found = cache.emplace(shape, plan).first;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudnnStatus_t status = cudnnConvolutionForward(
        handle(), &alpha, d.x, x, d.w, w, d.conv, found->second.algo,
        workspace, found->second.workspace_bytes, &beta, d.y, y);
    return status == CUDNN_STATUS_SUCCESS ? 0 : -2;
}
