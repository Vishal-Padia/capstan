# AGENTS.md

Instructions for any coding agent working in this repo.

## What this repo is

Capstan measures, verifies, and speeds up the rollout step of RL post-training for flow and diffusion models. Read [README.md](README.md) for scope. The current target is narrow: Qwen-Image OCR LoRA, verl-omni with MixGRPO, one GPU configuration. Do not build for anything outside that scope unless asked.

The three deliverables, in order:

1. Profiler: per-phase time for one training step.
2. Numerics check: rollout vs trainer log-prob drift, plus batch invariance.
3. One optimisation, measured by the profiler and gated by the numerics check.

## Numbers

- Every number in README.md or `bench/` comes from a real run. Never estimate, extrapolate, or fill in a placeholder with a plausible value.
- Each result records the GPU, card count, and verl-omni commit it ran on.
- Drift is reported per latent element. The importance ratio has the trainer log-prob in the numerator.
- The drift threshold is fixed before the run. Never change it after seeing results.
- A speedup is not a result without its drift number next to it.
- If a result contradicts a claim in README.md, change the claim to fit the result, not the other way round.

## Code

- Write the smallest change that works. No speculative abstractions, config for values that never change, or scaffolding for later.
- No comments that restate the code. No header or docstring blocks at the top of files. Comment only what the code cannot say: a non-obvious reason, a numerical subtlety, a known limitation.
- Reuse what verl-omni and vllm-omni already provide before writing your own.
- Non-trivial logic leaves one runnable check behind.
- Do not add dependencies without asking.


## Git and PRs

- Never commit or push unless the maintainer asks for it in that conversation.
- No `Co-Authored-By` trailers or any agent attribution in commits or PR bodies.
- One concern per PR. Keep each PR small enough to review line by line. Split large work into a chain of PRs.
- Branch from `main`. Never push to `main` directly.
- The maintainer reviews on GitHub and asks for changes in the agent session. Address each review comment in a new commit on the same branch, then report what changed for each comment.
