# capstan

Measure, verify, and speed up the rollout step of RL post-training for flow and diffusion models.

Capstan sits on top of the RL framework you already run. It does not own the algorithm. Today it owns three questions about one training step: where did the GPU time go, do rollout and trainer numerics still agree, and can rollout be made cheaper without breaking that agreement.

## The problem

RL post-training of a flow model is a loop: sample trajectories from the current policy (rollout), score them, recompute the per-step log-probabilities on the trainer, take a gradient step, push weights back to the sampler. Rollout is the expensive part: every image takes 10 denoising steps through the full model, two forward passes each with classifier-free guidance. How large a share of step time that is, the profiler measures.

Making rollout faster is only safe if the sampler and the trainer still compute the same log-probabilities for the same trajectory. They run different kernels, different fusions, sometimes different precisions. When they disagree, the samples come from a different policy than the one the trainer thinks it is updating, so training quietly goes off-policy. If the trainer skips its own recompute and uses the sampler's log-probs directly, the importance ratios themselves are wrong. Either way the policy can learn to exploit the gap instead of the reward. So a speedup number without a drift number is not a result.

Frameworks like verl-omni, verl, prime-rl, SkyRL and TRL own the algorithm. Capstan does not replace them.

## What is in this repo today

This is a proof of concept on one model, one framework, one GPU configuration.

| Deliverable | What it does | Output |
| --- | --- | --- |
| Profiler | Splits one MixGRPO step into rollout, reward, trainer recompute, trainer update, weight sync, and everything else, idle included. Inside rollout, splits text encoding, ODE steps, SDE-window steps and VAE decode. `capstan report` turns a recorded run into the tables. | [bench/profile.md](bench/profile.md) |
| Numerics check | Records rollout log-probs for the SDE-window steps, recomputes them on the trainer for the same trajectory, reports the drift. Also checks whether rollout log-probs change with batch size. Pass threshold, fixed before the first real run: \|ratio − 1\| ≤ 1e-4 (dimensionless, about 1e-4 nats of per-element log-prob drift) for at least 99% of (sample, SDE step) entries. 1e-4 is the recipe's own PPO clip window (`diffusion_loss.clip_ratio`): drift that large moves the ratio as far as one policy update is allowed to. | [bench/drift.md](bench/drift.md) |
| One optimisation | Chosen from the profile, not before it. Measured with the profiler, gated by the numerics check. | [bench/before_after.md](bench/before_after.md) |

Baseline it builds on: verl-omni, Flow-GRPO / MixGRPO, Qwen-Image, OCR LoRA recipe, 10-step trajectory, 2-step SDE window, random window strategy.

Run it. The steps are manual because installing verl-omni and downloading the models depend on the machine.

1. Install verl-omni at commit `f50117e` with its [installation guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/install.md), plus `uv pip install -e ".[ocr]"`, and build the OCR dataset into `$WORKSPACE/data/ocr/qwen_image/` with its `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py`.
2. Install Capstan into the same environment: `uv pip install -e .`
3. From the verl-omni repo root, run the MixGRPO recipe through Capstan. [`bench/run_mixgrpo.sh`](bench/run_mixgrpo.sh) is verl-omni's `examples/mixgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_mixgrpo.sh` with three changes: it runs through `capstan run`, turns on sampler log-probs (`calculate_log_probs=True`) so the drift check has both sides, and stops after 12 steps with no checkpoints, validation or W&B.

```bash
bash /path/to/capstan/bench/run_mixgrpo.sh
```

4. Turn the run into tables: `capstan report runs/mixgrpo`.

Recordings land in `runs/mixgrpo/`: one `phases_<pid>.jsonl` per process, and one `logprobs_step<step>_pid<pid>.npz` per training step.

## Results

One result so far, on Qwen-Image: batch invariance. The sampler's log-probs for the same trajectories differ by at most 1.19e-07 per element between batch size 1 and 16 (1x H100, step mode; details in [bench/drift.md](bench/drift.md)). That is one float32 rounding step, far under the 1e-4 drift threshold. The profile and the rollout-vs-trainer drift need the 4x H100 run, which has not happened yet.

Every real run reports, in `bench/`:

- rollout share of step time, and rollout and step time before and after the optimisation
- log-prob drift between BF16 rollout and BF16 trainer, as mean and max per-element absolute difference
- the importance ratio exp(trainer minus rollout) at p50 and p99, and how many entries are over the threshold
- batch invariance: the largest drift for the same trajectory at two batch sizes
- reward on a fixed prompt set, before and after
- GPU, VRAM, card count, and the verl-omni commit

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

Early proof of concept, single author. So far it has run end to end on one NVIDIA A10 (Lambda Cloud) with tiny random-weight Qwen-Image and Qwen3-VL models and verl-omni at `f50117e`: enough to check the recording, not to measure anything. The batch-invariance check has run on real Qwen-Image on 1x H100. The full training run, Qwen-Image OCR LoRA on 4x H100, has not happened yet. Untested: any other model, framework or GPU, full-parameter training, multi-node. The optimisation is one change and may not transfer.

## Why build this

RL post-training for image, video and audio models runs on hand-rolled sampling loops, and most of the GPU bill goes to generating samples. Nobody measures where that money goes, or checks that a faster sampler still trains the same policy. Capstan does both, so making rollout cheaper never costs training signal.

## Credits

Built on [verl-omni](https://github.com/verl-project/verl-omni) and [vllm-omni](https://github.com/vllm-project/vllm-omni). Algorithms: [Flow-GRPO](https://arxiv.org/abs/2505.05470) and [MixGRPO](https://arxiv.org/abs/2507.21802). Two pieces of the numerics check are borrowed from the vLLM team's train-inference consistency work ([Nov 2025](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference), [Oct 2026](https://vllm.ai/blog/2026-10-14-train-inference-consistency)): the batch-invariance check, and reporting the rollout-vs-trainer log-prob difference directly. Their work makes the two bitwise identical; Capstan only measures how far apart they are.

## License

[Apache 2.0](LICENSE)
