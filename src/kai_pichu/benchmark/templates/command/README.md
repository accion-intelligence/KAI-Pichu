# External C++ / CUDA executable

The default builds the CPU path with `c++`, so the contract can be tested
without a GPU. To run CUDA, set `options.compiler` to your `nvcc` and set
`compile_flags` to `[-O2, -arch=sm_XX]` for YOUR device, omitting `-x c++`.
Select the device using `CUDA_VISIBLE_DEVICES` before launching KAI.

This is process E2E latency: it includes startup, I/O, CUDA initialization,
allocation, copies, and computation. It is not kernel latency or a realistic
persistent-server benchmark. For an application's native kernel timer, parse
its structured output into `Observation.metrics["kernel_ms"]`, declare that
custom metric and its scope, and document its timer and synchronization.
PyTorch CUDA events cannot time GPU work in a different process.

Build products live in temporary directories and are cleaned up after the run.
The adapter uses argument arrays without a shell. Add all build inputs to
`implementation.files` when adapting a real project.
