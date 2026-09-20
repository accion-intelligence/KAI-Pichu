# Source and license notices

Third-party components in this repository retain their own licenses; the license texts are under [`licenses/`](licenses/). Framework code is licensed under the project's [Apache 2.0 license](LICENSE). Where a component was ported from another project, the source revision and file hashes are recorded so the attribution stays verifiable.

## Optimizer

Original optimizer components are Copyright (c) 2025 Zijian Zhang and retain [their MIT license](licenses/Optimizer-MIT.txt). The ported files, their upstream commit and hashes are recorded in [licenses/optimizer-source.json](licenses/optimizer-source.json).

## KAI Benchmark SDK

`src/kai_pichu/benchmark` is the KAI Benchmark SDK v1, including the paired buffer crossover and the CUDA event GC guard. It has no dependency on any other KAI repository and is covered by the project's [Apache 2.0 license](LICENSE).

## Profile query layer

The design of the profile queries and of `kai-ncu-reader` (narrow verbs over one report, a versioned JSON envelope, rows with stable keys) is inspired by [VeloQ](https://github.com/lucifer1004/veloq) (MIT License). No VeloQ code is included; the reader is written against NVIDIA's `ncu_report` API.

## AWQ linear-layer fusion example

`examples/awq_linear_fusion/kernels/dequantize.cuh` and the `dequantize_weights` kernel in `examples/awq_linear_fusion/kernels/dequantize_weights.cu` are copied from [casper-hansen/AutoAWQ_kernels](https://github.com/casper-hansen/AutoAWQ_kernels) (`awq_ext/quantization/`), MIT License, Copyright (c) 2023 Casper; the license text is in [licenses/AutoAWQ_kernels-MIT.txt](licenses/AutoAWQ_kernels-MIT.txt). The kernel credits compressa-ai/AutoAWQ, and `dequantize.cuh` is modified from NVIDIA FasterTransformer's `interleaved_numeric_conversion.h` (Apache-2.0); both files keep their original attribution comments. The launcher appended to the kernel file, `cublas_gemm.cu`, the adapter and the task definition are KAI code under the framework license. These notices do not relicense third-party material.
