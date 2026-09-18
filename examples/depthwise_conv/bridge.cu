#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "solution.cu"

// Fixed ABI shared by all workspaces. The adapter owns buffers, streams and
// timing. launch_depthwise_conv returns 0 on success or a negative code for
// arguments the implementation does not support; CUDA launch errors are
// reported after it. workspace is scratch memory with undefined contents.
extern "C" int kai_launch(
    const __half* x, const __half* w, __half* y, int n, int c, int h, int width,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    const int status = launch_depthwise_conv(x, w, y, n, c, h, width, workspace, workspace_bytes, stream);
    if (status != 0) {
        return status;
    }
    return static_cast<int>(cudaGetLastError());
}
