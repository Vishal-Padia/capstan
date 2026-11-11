# Drift

Threshold, fixed before the first real run: |ratio − 1| ≤ 1e-4 for at least 99% of (sample, SDE step) entries. Constants `RATIO_TOL` and `MAX_OVER_FRACTION` in `capstan/drift.py`.

| Step | Entries | Mean abs drift | Max abs drift | Ratio p50 | Ratio p99 | Over threshold |
| --- | --- | --- | --- | --- | --- | --- |

## Batch invariance

Sampler log-probs for the same 16 trajectories (one prompt, fixed per-sample seeds), generated with the engine allowed to batch 1 and then 16 requests. Qwen-Image, 1x H100 80 GB (Modal, driver 610), verl-omni `f50117e`, the MixGRPO recipe's rollout settings in step mode, run with [`batch_invariance.py`](batch_invariance.py). Step mode is required: in request mode, the recipe's default, the engine runs at `max_num_seqs=1` whatever is asked.

| Batch size a | Batch size b | Max per-element abs drift |
| --- | --- | --- |
| 1 | 16 | 1.19e-07 |

Both runs started each SDE window from identical latents, so they followed the same trajectories. A probe counting denoising forward passes (on the tiny model, same engine path) saw mostly full batches of 16 at `max_num_seqs=16` and only batches of 1 at `max_num_seqs=1`. 1.19e-07 is one float32 rounding step (2^-23).
