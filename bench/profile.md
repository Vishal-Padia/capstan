# Profile

## Baseline: Qwen-Image OCR LoRA, MixGRPO, request mode

4x NVIDIA H100 80 GB (Modal), verl-omni `81f3b65`, [`run_mixgrpo.sh`](run_mixgrpo.sh) as is: 32 prompts × 16 samples per step, 10 denoising steps, true CFG 4.0, SDE window of 2, rollout in request mode (one sample at a time per GPU). Run on Nov 26, 2026 with `modal run bench/modal_app.py::mixgrpo --name baseline`; 5 training steps, mean over steps 3–5 (the first 2 skipped as warm-up).

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
