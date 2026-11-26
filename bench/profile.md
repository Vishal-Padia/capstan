# Profile

## Baseline: Qwen-Image OCR LoRA, MixGRPO, request mode

4x NVIDIA H100 80 GB (Modal), verl-omni `81f3b65`, [`run_mixgrpo.sh`](run_mixgrpo.sh) as is: 32 prompts × 16 samples per step, 10 denoising steps, true CFG 4.0, SDE window of 2, rollout in request mode (one sample at a time per GPU). Run on Nov 26, 2026 with `modal run bench/modal_app.py::mixgrpo --name baseline`; 5 training steps, mean over steps 3–5 (the first 2 skipped as warm-up).

This host was slow: the same request-mode recipe ran at 533 s/step on another Modal 4x H100 host (below). Compare runs only within one container; see [before_after.md](before_after.md).

| Phase | Time per step (s) | Share of step |
| --- | --- | --- |
| rollout | 440.98 | 58.2% |
| reward | 108.65 | 14.3% |
| recompute | 36.54 | 4.8% |
| update | 148.65 | 19.6% |
| sync | 23.18 | 3.1% |
| everything else, idle included | 0.01 | 0.0% |
| step | 758.03 | 100% |

### Inside rollout

GPU time summed over the 4 rollout workers, per training step.

| Part | GPU seconds per step | Share of rollout GPU time | Calls per step |
| --- | --- | --- | --- |
| Text encode | 28.77 | 1.8% | 512 |
| ODE steps | 1241.08 | 77.8% | 4096 |
| SDE-window steps | 311.87 | 19.6% | 1024 |
| VAE decode | 13.50 | 0.8% | 512 |

Transformer forwards are 97.4% of rollout GPU time. An ODE step and an SDE step cost the same per call (0.303 s and 0.305 s, each a CFG pair of forwards), since only the scheduler step after the forward differs. The 1595 GPU-seconds over 4 workers fill about 90% of the 441 s rollout phase.

## Batched rollout vs request mode, one host

Request mode then step mode (`max_num_seqs=16`) in one container: 4x NVIDIA H100 80GB HBM3 (Modal, driver 620.70.06), verl-omni `81f3b65`, Nov 26, 2026, `modal run bench/modal_app.py::run --job mixgrpo-pair --name paired`. Means over steps 3–5.

| Phase | Request (s) | Share | Step mode (s) | Share |
| --- | --- | --- | --- | --- |
| rollout | 294.72 | 55.3% | 104.29 | 30.2% |
| reward | 72.40 | 13.6% | 76.82 | 22.3% |
| recompute | 30.25 | 5.7% | 30.40 | 8.8% |
| update | 121.20 | 22.7% | 119.94 | 34.7% |
| sync | 14.38 | 2.7% | 13.70 | 4.0% |
| step | 532.95 | 100% | 345.16 | 100% |

### Inside rollout

| Part | Request: GPU s/step | Calls | Step mode: GPU s/step | Calls |
| --- | --- | --- | --- | --- |
| Text encode | 17.67 | 512 | 16.20 | 512 |
| ODE steps | 877.80 | 4096 | 243.68 | 228 |
| SDE-window steps | 220.59 | 1024 | 102.35 | 96 |
| VAE decode | 12.25 | 512 | 11.89 | 512 |

Batching turns 5,120 transformer calls per step (each a CFG pair for one image) into 324 batched ones. A batch-1 ODE call costs 0.214 s on this host and 0.303 s on the baseline's; a batched call costs about the same on both (1.07 s and 1.04 s).
