# Task-Aware 12-Lead ECG Transmission Under Bandwidth and Noise Constraints

This is a **minimal research-code release** accompanying the manuscript
*"Task-Aware 12-Lead ECG Transmission Under Bandwidth and Noise
Constraints"* (Puyang Zhao, Jingxi Zhao, Huiyun Zhang), provided so
reviewers and readers can inspect and run the core method implementation
directly.

## What this method does

A shared per-lead 1D residual encoder maps each of a 12-lead ECG's leads
to a 32-dimensional latent embedding. A personalized, sample-specific
selector picks which `K` of the 12 leads to transmit. Each selected
lead's embedding is reinterpreted as 16 complex symbols, of which `m` are
actually sent (the rest dropped), each repeated `r` times for noise
protection, under a fixed per-lead budget `m * r = 16` complex channel
uses. Transmission is simulated over an AWGN or block-Rayleigh fading
channel; the receiver equalizes (Rayleigh) and combines the repeated
copies of each symbol, zero-fills whatever was never transmitted, and the
resulting per-lead latents are mean-pooled over the selected leads and
passed through a shared multilabel classification head.

- **Input:** a batch of raw 12-lead ECG windows, shape `[B, 12, T]`
  (`T=1000` at 100 Hz in the manuscript, i.e. 10 s).
- **Output:** multilabel diagnostic probabilities, shape
  `[B, num_classes]` (5 classes in the manuscript: NORM, MI, STTC, CD,
  HYP), plus the per-sample lead-selection mask and channel-use/energy
  diagnostics.

## What is included

```
src/
  encoder.py        shared per-lead 1D residual encoder
  lead_selector.py  sample-specific Top-K lead selection (proposed method)
  channel.py         AWGN / block-Rayleigh channel, equalization, fixed
                      N=16 symbol budget, m*r=16 repetition allocation/combining
  model.py            end-to-end assembly + multilabel classification head
examples/
  minimal_demo.py    runnable demo on synthetic tensors
configs/
  example_config.yaml
```

## What is intentionally NOT included

This is a method-implementation reference, not the full experimental
codebase. **Not included:** PTB-XL data or any data-download script,
trained model checkpoints, raw predictions or numerical-results JSON
files, the manuscript's figures or its LaTeX/PDF source, the full
training/evaluation/bootstrap pipeline, and the unit test suite used
during development. This repository does **not** reproduce every
manuscript result -- it demonstrates the method itself, runnably, on
synthetic input.

## Install

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10. `requirements.txt` pins `torch==2.13.0`; if you
need a CPU-only or a different CUDA build of PyTorch, install it first
per the instructions at https://pytorch.org/get-started/locally/ and then
`pip install -r requirements.txt` for the remaining (torch-independent)
dependencies.

## Run the demo

```bash
python examples/minimal_demo.py
```

This builds an untrained `TaskAwareECGModel` from
`configs/example_config.yaml`, runs a batch of random synthetic tensors
through it once under AWGN and once under block-Rayleigh fading, and
prints: the input shape, the selected lead indices per sample, the
channel-use/energy accounting, and the output probabilities. Because the
model is untrained (random initialization, no PTB-XL data), the printed
probabilities are illustrative only -- they demonstrate the pipeline's
shapes and control flow, not diagnostic performance.

## Not a clinical device

This is research code accompanying an academic manuscript. It is **not a
clinical decision-support system, not a diagnostic device, and has not
been validated, cleared, or approved for any clinical or diagnostic
use.** No component of this repository should be used to make, support,
or influence a real patient-care decision.

## License

MIT (see `LICENSE`) for the code in this repository. PTB-XL itself, if
you separately obtain it, is governed by PhysioNet's own license, not
this repository's.
