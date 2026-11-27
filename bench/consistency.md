# Consistency: where the rollout–trainer drift comes from

## One GPU, step 0, config by config

1x NVIDIA H100 80GB HBM3 (Modal, driver 620.70.06), verl-omni `81f3b65`, real Qwen-Image. Run on Nov 27, 2026 with `modal run bench/modal_app.py::run --job consistency --name c4-consistency`.

[`consistency.py`](consistency.py) runs the real rollout engine for 16 samples of one prompt with `sde_window_size=5`, so every sample records SDE log-probs at all five timesteps the recipe trains on. It then recomputes the trainer's log-probs for the same trajectories with diffusers' transformer in bf16 and verl-omni's own trainer step. There is no FSDP, no LoRA (step 0), and no training.

The rollout configs:

- `stock`: the recipe as is; FA3 from the kernels hub, torch.compile on.
- `eager`: `enforce_eager=True`, so no torch.compile.
- `reference`: eager plus `rollout_attn_backend=TORCH_SDPA`, which switches on vllm-omni's Qwen-Image `reference_precision` path. That path replaces the fused QK-norm+RoPE, AdaLN and modulation kernels with diffusers-matching code.

The trainer side runs with the recipe's `_flash_3_varlen_hub` attention, or `native` (torch SDPA).

Mean signed per-element drift (trainer minus rollout), ± standard error over 16 samples at t = 673:

| Rollout × trainer | t=1000 | 929 | 851 | 767 | 673 |
| --- | --- | --- | --- | --- | --- |
| stock × FA3 (the recipe) | −2.1e-6 | −7.0e-6 | −3.0e-5 | −4.8e-5 | −8.7e-5 ± 1e-5 |
| stock × native | −2.2e-6 | −6.1e-6 | −3.3e-5 | −5.9e-5 | −9.5e-5 ± 1e-5 |
| eager × FA3 | −1.3e-6 | −6.6e-6 | −2.9e-5 | −3.7e-5 | −5.5e-5 ± 6e-6 |
| eager × native | −1.4e-6 | −7.2e-6 | −1.3e-5 | −3.4e-5 | −5.6e-5 ± 6e-6 |
| reference × FA3 | −1.1e-6 | −3.7e-6 | −9.6e-6 | −3.3e-5 | −3.7e-5 ± 6e-6 |
| reference × native | −1.2e-6 | −2.3e-6 | −1.1e-5 | −2.5e-5 | −4.9e-5 ± 1e-5 |

- **It reproduces at step 0 on one GPU.** Stock gives −8.7e-5 at t = 673; the 4x H100 training run gave −8.6e-5 ([before_after.md](before_after.md)). The drift comes from the two transformer implementations, not from LoRA, FSDP or training.
- **torch.compile on the rollout side is about 40% of it** (−8.7e-5 → −5.5e-5). Turning it off costs rollout speed, so it isn't a fix until it's timed.
- **The trainer's attention backend makes no measurable difference.**
- **vllm-omni's reference mode leaves about half of it**, not distinguishable from plain eager at 16 samples.
- **The growth with denoising comes from the scheduler.** For the recipe's SDE, a velocity mismatch Δv between the two sides gives a per-element drift of about −1.48 · |dt| · (1−σ)/(a²σ) · mean(Δv²), where a is the noise level (1.2). Dividing the stock drift by |dt|·(1−σ)/σ gives a roughly constant value from t = 929 to t = 673. So the model mismatch is about the same at every step, and the falling noise scale amplifies it. The drift is always negative because the mismatch enters squared.
