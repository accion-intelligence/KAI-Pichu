# Stateful workload

This dependency-free CPU example demonstrates the lifecycle needed by recurrent
inference and training adapters. Each `run` executes a complete sequence.
`reset` restores the initial state before every invocation. Verification checks
all outputs AND the final state, with invalid probes for both.

For real training, include optimizer state, RNG state, gradients, and any graph
capture/replay semantics required by your task. Reset any mutable state hidden
in the loaded implementation too. The SDK cannot discover that state for you.
