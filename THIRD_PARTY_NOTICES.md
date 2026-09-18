# Source and license notices

This documentation bundle reflects the [source repository notices at commit 451a839](https://github.com/accion-intelligence/KAI-Core/blob/451a839fdea43da3513cb33c99075782477fa8ce/THIRD_PARTY_NOTICES.md). It contains documentation and license texts, not the executable source tree. Source-history references remain pinned to that repository; this page does not replace or change its original notices.

## Optimizer

Original optimizer components are Copyright (c) 2025 Zijian Zhang and retain [their MIT license](licenses/Optimizer-MIT.txt). Source revision and hashes are recorded in [the source history](https://github.com/accion-intelligence/KAI-Core/blob/451a839fdea43da3513cb33c99075782477fa8ce/licenses/optimizer-source.json).

## KAI Benchmark SDK

The source repository's `src/kai_core/benchmark` is a standalone copy of the KAI Benchmark SDK v1, including the paired buffer crossover and CUDA event GC guard. Public imports and CLI branding use `kai_core`. The parent KAI repository is not a runtime dependency. Framework changes use the project's [Apache 2.0 license](LICENSE).

## AWQ linear-layer fusion example

`examples/awq_linear_fusion/kernels/dequantize.cuh` and the `dequantize_weights` kernel in `examples/awq_linear_fusion/kernels/dequantize_weights.cu` are copied from [casper-hansen/AutoAWQ_kernels](https://github.com/casper-hansen/AutoAWQ_kernels) (`awq_ext/quantization/`), MIT License, Copyright (c) 2023 Casper; the license text is in [licenses/AutoAWQ_kernels-MIT.txt](licenses/AutoAWQ_kernels-MIT.txt). The kernel credits compressa-ai/AutoAWQ, and `dequantize.cuh` is modified from NVIDIA FasterTransformer's `interleaved_numeric_conversion.h` (Apache-2.0); both files keep their original attribution comments. The launcher appended to the kernel file, `cublas_gemm.cu`, the adapter and the task definition are KAI code under the framework license. These notices do not relicense third-party material.
