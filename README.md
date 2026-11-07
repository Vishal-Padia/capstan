# capstan

Measure, verify, and speed up the rollout step of RL post-training for flow and diffusion models.

Capstan sits on top of the RL framework you already run. It does not own the algorithm. Today it owns three questions about one training step: where did the GPU time go, do rollout and trainer numerics still agree, and can rollout be made cheaper without breaking that agreement.

## The problem

RL post-training of a flow model is a loop: sample trajectories from the current policy (rollout), score them, recompute the per-step log-probabilities on the trainer, take a gradient step, push weights back to the sampler. Almost all of the GPU time is in the first part.

In the run below, rollout was **[XX]%** of step time on **[GPU]**.

Making rollout faster is only safe if the sampler and the trainer still compute the same log-probabilities for the same trajectory. They run different kernels, different fusions, sometimes different precisions. When they disagree, the importance ratios the loss depends on are wrong, and the policy can learn to exploit the gap instead of the reward. So a speedup number without a drift number is not a result.

Frameworks like verl-omni, verl, prime-rl, SkyRL and TRL own the algorithm. Capstan does not replace them.

## What is in this repo today

This is a proof of concept on one model, one framework, one GPU configuration.

| Deliverable | What it does | Output |
| --- | --- | --- |
| Profiler | Splits one MixGRPO step into rollout, reward, trainer recompute and update, weight sync, and idle time. Inside rollout, splits ODE steps from SDE-window steps. | [bench/profile.md](bench/profile.md) |
| Numerics check | Records rollout log-probs for the SDE-window steps, recomputes them on the trainer for the same trajectory, reports the drift. Also checks whether rollout log-probs change with batch size. Pass threshold, fixed before the first run: [threshold, units]. | [bench/drift.md](bench/drift.md) |
| One optimisation | [Candidate B or C, filled in after profiling]. Measured with the profiler, gated by the numerics check. | [bench/before_after.md](bench/before_after.md) |

Baseline it builds on: verl-omni, Flow-GRPO / MixGRPO, Qwen-Image, OCR LoRA recipe, 10-step trajectory, 2-step SDE window, random window strategy.

Run it:

```
[one command, or the exact manual steps and why they are manual]
```

## Results

All numbers below come from the runs in `bench/`. Nothing here is estimated.

| Measure | Value |
| --- | --- |
| Rollout share of step time | [XX]% |
| Rollout time per step, before | [X] s |
| Rollout time per step, after | [X] s |
| Speedup on rollout | [X]x |
| Step time, before and after | [X] s, [X] s |
| Log-prob drift, BF16 rollout vs BF16 trainer, per-element mean abs | [X] |
| Log-prob drift, max over samples and steps, per-element abs | [X] |
| Importance ratio exp(trainer minus rollout), p50 / p99 | [X] / [X] |
| Steps over the drift threshold | [X] of [N] |
| Batch-invariance: same trajectory, batch size [a] vs [b], max drift | [X] |
| Reward on the fixed prompt set, before and after | [X], [X] |
| GPU, VRAM, count | [model], [X] GB, [N] |
| verl-omni commit | [hash] |

Two notes on reading the numbers:

- Drift is reported per latent element. verl-omni's per-step log-prob is already a mean over the latent, not a sum, so the drift of one sample at one SDE-window step is exactly what enters the importance ratio: exp(trainer minus rollout), trainer in the numerator. "Max" means the largest drift over all samples and SDE-window steps; the per-element differences inside one latent are never materialised.
- Weight sync looks negligible in the profile because this is a LoRA run. That is a LoRA artifact, not a claim that sync is cheap for full-parameter training.

## Scope and roadmap

Listing something here means roadmap, not working code.

- Today: flow and diffusion image models, one framework integration (verl-omni), one GPU configuration
- Next: diffusion and flow video models
- Next: streaming audio and TTS models
- Next: state-space models
- Next: FP8 and FP4 rollout paths gated by the same numerics check
- Next: a rollout engine and weight-sync path of its own; today both are verl-omni's and vllm-omni's
- Next: adapters for verl, prime-rl, SkyRL and TRL

## Non-goals

- Not a new RL algorithm
- Not a replacement for verl-omni, verl, prime-rl, SkyRL or TRL
- Not an inference server for autoregressive text LLMs; vLLM and SGLang already cover that, including their RL weight-sync and consistency work

## Status and limitations

Early proof of concept, single author. Tested on [GPU] with [N] cards, verl-omni at [commit], Qwen-Image OCR LoRA only. Untested: any other model, any other framework, any other GPU, full-parameter training, multi-node. The optimisation in Phase 3 is one change and may not transfer.

## Why build this

[Two or three lines, in your own words, written last.]

## Credits

Built on [verl-omni](https://github.com/verl-project/verl-omni) and [vllm-omni](https://github.com/vllm-project/vllm-omni). Algorithms: [Flow-GRPO](https://arxiv.org/abs/2505.05470) and [MixGRPO](https://arxiv.org/abs/2507.21802). Two pieces of the numerics check are borrowed from the vLLM team's train-inference consistency work ([Nov 2025](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference), [Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)): the batch-invariance check, and reporting the rollout-vs-trainer log-prob difference directly. Their work makes the two bitwise identical; Capstan only measures how far apart they are.

## License

[Apache 2.0](LICENSE)
