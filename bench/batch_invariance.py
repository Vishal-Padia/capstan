import argparse
import os
import sys

import numpy as np

from capstan.drift import drift_stats

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)
PROMPT = 'A wooden street sign on a pole that reads "CAPSTAN" in white letters.'

# The rollout settings of verl-omni's MixGRPO Qwen-Image OCR recipe, with sampler log-probs on.
RECIPE = [
    "actor_rollout_ref.model.algorithm=mix_grpo",
    "actor_rollout_ref.rollout.name=vllm_omni",
    "actor_rollout_ref.rollout.mode=async",
    "actor_rollout_ref.rollout.n=16",
    "actor_rollout_ref.rollout.calculate_log_probs=True",
    # Request mode, the recipe's default, runs the engine at max_num_seqs=1 whatever is asked, so it never batches.
    # Step mode honours max_num_seqs, which is what makes this a batch-size comparison at all.
    "actor_rollout_ref.rollout.step_execution=True",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.agent.num_workers=1",
    "actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent",
    "actor_rollout_ref.rollout.pipeline.num_inference_steps=10",
    "actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0",
    "actor_rollout_ref.rollout.pipeline.max_sequence_length=256",
    "actor_rollout_ref.rollout.algo.sample_strategy=random",
    "actor_rollout_ref.rollout.algo.sde_window_seed=42",
    "actor_rollout_ref.rollout.algo.sde_window_size=2",
    "actor_rollout_ref.rollout.algo.sde_window_range=[0,5]",
    "actor_rollout_ref.rollout.algo.noise_level=1.2",
    "actor_rollout_ref.rollout.algo.sde_type=sde",
    "data.max_prompt_length=256",
    "reward.reward_manager.name=image",
    "trainer.n_gpus_per_node=1",
    "trainer.nnodes=1",
]


def generate(model, batch, overrides, out):
    import ray
    import verl_omni
    from hydra import compose, initialize_config_dir
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    from verl.protocol import DataProto
    from verl.workers.rollout.llm_server import LLMServerManager
    from verl_omni.agent_loop import DiffusionAgentLoopWorker

    config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                *RECIPE,
                f"actor_rollout_ref.model.path={model}",
                f"actor_rollout_ref.model.tokenizer_path={model}/tokenizer",
                f"actor_rollout_ref.rollout.max_num_seqs={batch}",
                *overrides,
            ],
        )

    n = config.actor_rollout_ref.rollout.n
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PROMPT}]
    negative = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": " "}]
    prompts = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([messages] * n),
            "raw_negative_prompt": np.array([negative] * n),
            "data_source": np.array(["jpeg_compressibility"] * n),
            "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * n),
            # Fixed per-sample seeds: sample i draws the same noise whatever else is batched with it.
            "_rollout_seed_global_idx": np.arange(n, dtype=np.int64),
        },
        meta_info={"global_steps": 1, "rollout_seed": 42},
    )

    ray.init()
    try:
        AgentLoopManager.agent_loop_workers_class = ray.remote(DiffusionAgentLoopWorker)
        servers = LLMServerManager.create(config=config)
        manager = AgentLoopManager.create(config=config, llm_client=servers.get_client())
        result = manager.generate_sequences(prompts=prompts)
    finally:
        ray.shutdown()
    np.savez(
        out,
        batch=batch,
        log_probs=result.batch["rollout_log_probs"].float().cpu().numpy(),
        window_start=result.batch["all_latents"][:, 0].float().cpu().numpy(),
    )


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    same_start = np.array_equal(a["window_start"], b["window_start"])
    drift = drift_stats(a["log_probs"], b["log_probs"])["max_abs"]
    print("| Batch size a | Batch size b | Max per-element abs drift |")
    print("| --- | --- | --- |")
    print(f"| {int(a['batch'])} | {int(b['batch'])} | {drift:.3g} |")
    if not same_start:
        # Different noise at the window start means the runs did not follow the same trajectories,
        # so the drift above mixes RNG with numerics and is not a batch-invariance number.
        print("\nWindow-start latents differ: not the same trajectories.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Sampler log-probs for the same trajectories at two batch sizes.")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="run one batch size; extra args are verl-omni overrides")
    gen.add_argument("--model", required=True)
    gen.add_argument("--batch", type=int, required=True, help="requests the engine may batch together")
    gen.add_argument("--out", required=True)
    cmp = commands.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args, overrides = parser.parse_known_args()
    if args.command == "generate":
        generate(args.model, args.batch, overrides, args.out)
    elif overrides:
        parser.error(f"unrecognized arguments: {' '.join(overrides)}")
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
