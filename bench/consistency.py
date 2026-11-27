import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from batch_invariance import rollout

from capstan.drift import drift_stats

# A window of 5 inside the recipe's range [0, 5] can only start at 0, so every sample records SDE log-probs at all
# five timesteps the recipe ever trains on (1000 down to 673) instead of a random pair of them.
WINDOW = ["actor_rollout_ref.rollout.algo.sde_window_size=5"]
NOISE_LEVEL, SDE_TYPE, TRUE_CFG, STEPS = 1.2, "sde", 4.0, 10


def generate(model, overrides, out):
    result = rollout(model, 16, [*WINDOW, *overrides])
    torch.save({k: v.cpu() for k, v in result.batch.items() if isinstance(v, torch.Tensor)}, out)
    print("saved", out, {k: tuple(v.shape) for k, v in result.batch.items()})


@torch.no_grad()
def recompute(model, traj, attn_backend, out, height=512, width=512):
    from diffusers import QwenImageTransformer2DModel
    from verl_omni.pipelines.qwen_image_flow_grpo.common import QWEN_IMAGE_VAE_SCALE_FACTOR, build_img_shapes
    from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import (
        QwenImage,
        _build_qwen_image_scheduler,
        _configure_qwen_image_scheduler,
    )

    d = torch.load(traj)
    module = QwenImageTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.bfloat16)
    module = module.cuda().eval()
    module.set_attention_backend(attn_backend)
    # FSDP2 casts floating forward inputs to bf16 unless the model declares fp32 islands (verl-omni's
    # _fsdp_param_dtype); do the same so the trainer side sees what the real trainer sees.
    cast = not getattr(module, "_keep_in_fp32_modules", None)
    print("fp32 islands:", getattr(module, "_keep_in_fp32_modules", None), "-> cast inputs to bf16:", cast)
    scheduler = _build_qwen_image_scheduler(model)
    _configure_qwen_image_scheduler(scheduler, height=height, width=width, num_inference_steps=STEPS, device="cuda")
    config = SimpleNamespace(
        pipeline=SimpleNamespace(true_cfg_scale=TRUE_CFG),
        algo=SimpleNamespace(noise_level=NOISE_LEVEL, sde_type=SDE_TYPE),
    )

    def bf16(x):
        return x.to(torch.bfloat16) if cast and x.is_floating_point() else x

    latents, timesteps = d["all_latents"].cuda(), d["all_timesteps"].cuda()
    img_shapes = build_img_shapes(height, width, latents.shape[0], QWEN_IMAGE_VAE_SCALE_FACTOR)
    trainer = []
    for step in range(timesteps.shape[1]):
        inputs = []
        for prefix in ("", "negative_"):
            inputs.append(
                {
                    "hidden_states": bf16(latents[:, step]),
                    "timestep": bf16(timesteps[:, step] / 1000.0),
                    "guidance": None,
                    "encoder_hidden_states": bf16(d[f"{prefix}prompt_embeds"].cuda()),
                    "encoder_hidden_states_mask": d[f"{prefix}prompt_embeds_mask"].cuda(),
                    "img_shapes": img_shapes,
                    "return_dict": False,
                }
            )
        log_prob, *_ = QwenImage.forward_and_sample_previous_step(
            module, scheduler, config, inputs[0], inputs[1], {"all_latents": latents, "all_timesteps": timesteps}, step
        )
        trainer.append(log_prob.float().cpu())
    np.savez(
        out,
        rollout=d["rollout_log_probs"].float().numpy(),
        trainer=torch.stack(trainer, dim=1).numpy(),
        timesteps=d["all_timesteps"].float().numpy(),
    )


def table(paths):
    print("| Run | Timestep | Mean signed drift | Max abs drift | Entries over 1e-4 |")
    print("| --- | --- | --- | --- | --- |")
    for path in paths:
        a = np.load(path)
        t = np.round(a["timesteps"].astype(np.float64), 2)
        for ts in sorted(set(t.ravel().tolist()), reverse=True):
            m = t == ts
            s = drift_stats(a["rollout"][m], a["trainer"][m])
            signed = float(np.mean(a["trainer"][m].astype(np.float64) - a["rollout"][m]))
            print(f"| {Path(path).stem} | {ts:g} | {signed:.2g} | {s['max_abs']:.3g} | {s['over_tol']} of {s['n']} |")


def main():
    parser = argparse.ArgumentParser(description="Rollout vs trainer log-probs on one GPU, one config at a time.")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="rollout trajectories; extra args are verl-omni overrides")
    gen.add_argument("--model", required=True)
    gen.add_argument("--out", required=True)
    rec = commands.add_parser("recompute", help="trainer log-probs for saved trajectories with diffusers")
    rec.add_argument("--model", required=True)
    rec.add_argument("--traj", required=True)
    rec.add_argument("--attn-backend", required=True)
    rec.add_argument("--out", required=True)
    tab = commands.add_parser("table")
    tab.add_argument("paths", nargs="+")
    args, overrides = parser.parse_known_args()
    if args.command == "generate":
        generate(args.model, overrides, args.out)
    elif overrides:
        parser.error(f"unrecognized arguments: {' '.join(overrides)}")
    elif args.command == "recompute":
        recompute(args.model, args.traj, args.attn_backend, args.out)
    else:
        table(args.paths)


if __name__ == "__main__":
    main()
