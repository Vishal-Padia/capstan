import argparse
import os

import torch

STEP = 4  # t = 673, the timestep with the most drift
POINTS = ["txt_in", "time_text_embed", *[f"transformer_blocks.{i}" for i in range(60)], "proj_out"]


# One sample's CFG pair, as each side sees it: the rollout encodes prompts at their own length and vllm-omni drops an
# all-ones mask, while verl-omni pads the trainer's prompt embeddings to a fixed length behind a mask.
def calls(traj, padded):
    d = torch.load(traj)
    x = d["all_latents"][:1, STEP].cuda().to(torch.bfloat16)
    t = d["all_timesteps"][:1, STEP].cuda().float() / 1000
    out = []
    for prefix in ("", "negative_"):
        emb = d[f"{prefix}prompt_embeds"][:1].cuda().to(torch.bfloat16)
        mask = d[f"{prefix}prompt_embeds_mask"][:1].cuda()
        if not padded:
            n = int(mask.sum())
            emb, mask = emb[:, :n], mask[:, :n]
        out.append(
            {
                "hidden_states": x,
                "timestep": t,
                "encoder_hidden_states": emb,
                "encoder_hidden_states_mask": mask,
                "img_shapes": [[(1, 32, 32)]],
                "return_dict": False,
            }
        )
    return out


@torch.no_grad()
def dump(model, inputs, out):
    acts, current = {}, {}

    def hook(name):
        def save(module, args, output):
            key = f"{current['call']}/{name}"
            if isinstance(output, tuple):
                acts[f"{key}/txt"] = output[0].float().cpu()
                output = output[1]
            acts[key] = output.float().cpu()

        return save

    for name in POINTS:
        model.get_submodule(name).register_forward_hook(hook(name))
    for current["call"], kwargs in zip(("pos", "neg"), inputs, strict=True):
        model(**kwargs)
    torch.save(acts, out)
    print("saved", out, len(acts), "tensors")


def dump_vllm(model_path, backend, traj, out):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    config = OmniDiffusionConfig.from_kwargs(
        model=model_path,
        model_class_name="QwenImagePipeline",
        dtype=torch.bfloat16,
        enforce_eager=True,
        diffusion_attention_config={"default": {"backend": backend}},
    )
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=config)
    transformer = worker.model_runner.pipeline.transformer
    print("reference_precision:", transformer.reference_precision)
    with set_forward_context(vllm_config=worker.vllm_config, omni_diffusion_config=config):
        inputs = calls(traj, padded=False)
        for kwargs in inputs:
            kwargs["txt_seq_lens"] = [int(kwargs["encoder_hidden_states_mask"].sum())]
        dump(transformer, inputs, out)


def dump_diffusers(model_path, backend, traj, padded, out):
    from diffusers import QwenImageTransformer2DModel

    module = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    module = module.cuda().eval()
    module.set_attention_backend(backend)
    dump(module, calls(traj, padded), out)


# Text-shaped tensors are compared over the unpadded side's real tokens; padding rows carry nothing.
def rel(a, b):
    a, b = a.double(), b.double()[:, : a.shape[1]]
    return float((a - b).norm() / b.norm())


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    print(f"| Point | pos img | neg img | pos txt | ({a_path} vs {b_path}) |")
    print("| --- | --- | --- | --- | --- |")
    for name in POINTS:
        row = [rel(a[f"pos/{name}"], b[f"pos/{name}"]), rel(a[f"neg/{name}"], b[f"neg/{name}"])]
        txt = f"pos/{name}/txt"
        if txt in a:
            row.append(rel(a[txt], b[txt]))
        print(f"| {name} | " + " | ".join(f"{r:.2e}" for r in row) + " |")

    # apply_true_cfg, the combine both sides run; its error is what the SDE step turns into drift.
    def cfg(x):
        pos, neg = x["pos/proj_out"].double(), x["neg/proj_out"].double()
        comb = neg + 4.0 * (pos - neg)
        return comb * (pos.norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))

    dv = cfg(a) - cfg(b)
    print(f"\nAfter CFG: RMS dv {float(dv.pow(2).mean().sqrt()):.3e}, rel {rel(cfg(a), cfg(b)):.2e}")


def main():
    parser = argparse.ArgumentParser(description="Per-block activations of both Qwen-Image transformers.")
    commands = parser.add_subparsers(dest="command", required=True)
    for side in ("vllm", "diffusers"):
        p = commands.add_parser(side)
        p.add_argument("--model", required=True)
        p.add_argument("--backend", required=True)
        p.add_argument("--traj", required=True)
        p.add_argument("--out", required=True)
        if side == "diffusers":
            p.add_argument("--unpadded", action="store_true")
    cmp = commands.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args = parser.parse_args()
    if args.command == "vllm":
        dump_vllm(args.model, args.backend, args.traj, args.out)
    elif args.command == "diffusers":
        dump_diffusers(args.model, args.backend, args.traj, not args.unpadded, args.out)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
