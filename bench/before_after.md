# Before and after

## Batched rollout: request mode → step mode with `max_num_seqs=16`

Both runs in **one container on one host**: 4x NVIDIA H100 80GB HBM3 (Modal, driver 620.70.06), verl-omni `81f3b65`, the stock MixGRPO Qwen-Image OCR LoRA recipe, the same seeded SDE windows. 5 training steps each, means over steps 3–5. Run on Nov 26, 2026 with `modal run bench/modal_app.py::run --job mixgrpo-pair --name paired`.

The change is two overrides: `actor_rollout_ref.rollout.step_execution=True` and `actor_rollout_ref.rollout.max_num_seqs=16`. In request mode, the recipe's default, each GPU generates one image at a time; step mode batches up to 16 per transformer forward.

| Measure | Before (request) | After (step, 16) |
| --- | --- | --- |
| Step time (s) | 532.95 | 345.16 (**1.54x**) |
| Rollout time per step (s) | 294.72 | 104.29 (**2.83x**) |
| Reward / recompute / update / sync (s) | 72.40 / 30.25 / 121.20 / 14.38 | 76.82 / 30.40 / 119.94 / 13.70 |
| Per-element mean abs drift | 3.15e-05 | 3.1e-05 |
| Per-element max abs drift | 3.32e-04 | 3.02e-04 |
| Importance ratio p50 / p99 | 0.999986 / 1.000032 | 0.999987 / 1.000031 |
| Entries over the drift threshold | 321 of 5120 (fail) | 304 of 5120 (fail) |
| Mean training reward, steps 1–5 | 0.600, 0.816, 0.771, 0.630, 0.761 | 0.586, 0.816, 0.783, 0.650, 0.755 |

- **Speedup:** rollout 2.83x, the whole step 1.54x. The phases batching doesn't touch are unchanged. After batching, rollout is 30% of the step and the trainer update (35%) is the largest phase.
- **Drift:** unchanged. Both runs fail the threshold by the same amount, timestep by timestep (see [drift.md](drift.md)). Batching doesn't add drift, consistent with the batch-invariance result. The failure is the baseline's sampler–trainer mismatch, not batching.
- **Reward:** the per-step mean reward of the training batches (the same prompts at the same step in both runs) moves the same way in both. This is a sanity check that training behaves alike, not a held-out evaluation, which hasn't been run.
- **Why one host:** an earlier comparison across two separate containers showed 4.45x rollout and 2.29x end to end. The identical request-mode recipe takes 758 s/step on one Modal 4x H100 host and 533 s on another, so a before/after is only fair within one container.

## No torch.compile on the rollout side: `enforce_eager=True`

Both runs in **one container on one host**: 4x NVIDIA H100 80GB HBM3 (Modal, driver 620.70.06), verl-omni `81f3b65`. Both use batched rollout (step mode, `max_num_seqs=16`), and both drew the same seeded SDE windows. 5 training steps each, means over steps 3–5. Run on Nov 28, 2026 with the `eager-pair` job (`mixgrpo_pair(name, 5, "eager")`), which starts a run like `modal run bench/modal_app.py::run --job eager-pair --name c4-eager-pair`. This run was started on the deployed app (`modal deploy bench/modal_app.py`, then `mixgrpo_pair.spawn`) because the local client kept dropping.

The change is one override, `actor_rollout_ref.rollout.enforce_eager=True`. It turns off vllm-omni's torch.compile for the diffusion rollout. The trainer is unchanged.

| Measure | Before (compile on) | After (compile off) |
| --- | --- | --- |
| Step time (s) | 363.48 | 358.33 |
| Rollout time per step (s) | 104.71 | 102.82 |
| Reward / recompute / update / sync (s) | 82.41 / 30.45 / 134.04 / 11.85 | 84.75 / 30.41 / 128.88 / 11.46 |
| Per-element mean abs drift | 3.19e-05 | 2.08e-05 |
| Per-element max abs drift | 3.40e-04 | 3.84e-04 |
| Importance ratio p50 / p99 | 0.999987 / 1.000034 | 0.999993 / 1.000031 |
| Entries over the drift threshold | 340 of 5120 (6.6%, fail) | 105 of 5120 (2.05%, fail) |
| Signed drift at t = 673, entries over | −9.0e-05, 209 of 512 | −5.1e-05, 77 of 512 |
| Mean training reward, steps 1–5 | 0.600, 0.821, 0.766, 0.637, 0.777 | 0.595, 0.814, 0.773, 0.623, 0.757 |

- **Speed:** no cost. Rollout is 1.8% faster without compile, which is within run-to-run noise. Batched forwards spend their time in GEMMs and FA3 attention, which compile doesn't change.
- **Drift:** 3.2x fewer entries over the threshold, and the signed drift at t = 673 falls by 43%. That matches the one-GPU result in [consistency.md](consistency.md) (−8.7e-5 → −5.5e-5).
- **It still fails the threshold:** 2.05% of entries are over, against the 1% allowed. The rest of the mismatch comes from running two different bf16 transformer implementations, which no config change removes (see [consistency.md](consistency.md)).
- **Reward:** moves the same way in both runs. This is a sanity check, not an evaluation.
