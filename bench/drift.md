# Drift

Threshold, fixed before the first real run: |ratio − 1| ≤ 1e-4 for at least 99% of (sample, SDE step) entries. Constants `RATIO_TOL` and `MAX_OVER_FRACTION` in `capstan/drift.py`.

## Baseline: Qwen-Image OCR LoRA, MixGRPO, request mode

The same run as the [profile](profile.md): 4x NVIDIA H100 80 GB (Modal), verl-omni `81f3b65`, 5 training steps, 512 samples × 2 SDE-window steps per step. Drift is the trainer's recomputed log-prob minus the rollout's, per latent element, before any update in that step.

**Failed:** 370 of 5120 entries (7.2%) have |ratio − 1| > 1e-4; the threshold allows 1%.

| Step | Entries | Mean abs drift | Max abs drift | Ratio p50 | Ratio p99 | Over threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8.75e-06 | 5.33e-05 | 0.999997 | 1.000021 | 0 |
| 2 | 1024 | 7.26e-05 | 0.00058 | 0.999940 | 1.000041 | 262 |
| 3 | 1024 | 3.88e-05 | 0.000249 | 0.999973 | 1.000033 | 61 |
| 4 | 1024 | 7.96e-06 | 4.54e-05 | 0.999998 | 1.000022 | 0 |
| 5 | 1024 | 3.66e-05 | 0.000386 | 0.999974 | 1.000037 | 47 |
| all | 5120 | 3.3e-05 | 0.00058 | 0.999987 | 1.000032 | 370 |

The SDE window moves each step (`sample_strategy=random`, seeded per step), and the drift follows the timestep, not the training step. Split by SDE timestep, from the same log-prob files:

| Timestep t | Steps | Mean drift (signed) | Max abs drift | Over threshold |
| --- | --- | --- | --- | --- |
| 1000 | 1, 4 | −1.6e-6, −6.6e-7 | 1.8e-5 | 0 of 1024 |
| 928.97 | 1, 4 | −7.6e-6, −6.4e-6 | 5.3e-5 | 0 of 1024 |
| 851.48 | 3, 5 | −1.7e-5, −1.6e-5 | 2.0e-4 | 4 of 1024 |
| 766.62 | 2, 3, 5 | −4.3e-5, −5.1e-5, −4.6e-5 | 5.0e-4 | 142 of 1536 |
| 673.28 | 2 | −9.5e-5 | 5.8e-4 | 224 of 512 |

- Drift grows as denoising proceeds: negligible at the high-noise steps, past the recipe's own PPO clip (`clip_ratio=1e-4`) for 44% of samples at t = 673.
- It is always negative: the trainer scores the rollout's samples slightly lower than the sampler did.
- Steps 1 (LoRA still zero) and 4 (after three updates) share a window and show the same small drift, so it tracks the timestep, not training progress.
- This is request mode on a verl-omni commit that includes the step-mode timestep fix, so it is the plain mismatch between vllm-omni's rollout kernels and the diffusers trainer, not that bug. verl-omni's own measurements in [#280](https://github.com/verl-project/verl-omni/issues/280) found the same timestep dependence.


## Batched rollout vs request mode, one host

The paired run in [before_after.md](before_after.md): request mode then step mode (`max_num_seqs=16`) in one 4x H100 container, verl-omni `81f3b65`. **Both fail**: request 321 of 5120 entries over, step mode 304 of 5120. Batching leaves the drift where it was, timestep by timestep:

| Timestep t | Request: over | Request: signed mean | Step mode: over | Step mode: signed mean |
| --- | --- | --- | --- | --- |
| 1000 | 0 of 1024 | −1.1e-6 | 0 of 1024 | −1.1e-6 |
| 928.97 | 0 of 1024 | −6.9e-6 | 0 of 1024 | −6.5e-6 |
| 851.48 | 1 of 1024 | −1.7e-5 | 4 of 1024 | −1.6e-5 |
| 766.62 | 124 of 1536 | −4.5e-5 | 116 of 1536 | −4.5e-5 |
| 673.28 | 196 of 512 | −8.6e-5 | 184 of 512 | −8.5e-5 |

## Batch invariance

Sampler log-probs for the same 16 trajectories (one prompt, fixed per-sample seeds), generated with the engine allowed to batch 1 and then 16 requests. Qwen-Image, 1x H100 80 GB (Modal, driver 610), verl-omni `f50117e`, the MixGRPO recipe's rollout settings in step mode, run with [`batch_invariance.py`](batch_invariance.py). Step mode is required: in request mode, the recipe's default, the engine runs at `max_num_seqs=1` whatever is asked.

| Batch size a | Batch size b | Max per-element abs drift |
| --- | --- | --- |
| 1 | 16 | 1.19e-07 |

Both runs started each SDE window from identical latents, so they followed the same trajectories. A probe counting denoising forward passes (on the tiny model, same engine path) saw mostly full batches of 16 at `max_num_seqs=16` and only batches of 1 at `max_num_seqs=1`. 1.19e-07 is one float32 rounding step (2^-23). Re-run on Nov 17 at verl-omni `81f3b65` (vllm 0.33) with `modal run bench/modal_app.py::batch_invariance`: the same 1.19e-07.
