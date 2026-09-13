# Stateless PyTorch operator

Requires PyTorch in the existing environment. The default runs on CPU for
onboarding. For a CUDA operator, set `options.device: cuda:0`,
`measurement.timer: cuda_event`, and `measurement.device: 0` together. Use the
host-visible device index for this process. Change the measurement boundary to
describe GPU stream timing when using CUDA events.

`solution.py` is the only editable implementation file. The adapter checks
shape, dtype, device, and numerical agreement with an independent addition.
Search/acceptance also cover tail lengths. Replace those cases and tolerances
with requirements from your real project. If you enable `measurement.calibration`,
the small CPU example may fail it; that is a measurement finding, not a reason to
silently loosen the threshold.
