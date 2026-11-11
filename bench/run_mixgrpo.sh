#!/usr/bin/env bash
set -euxo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
OUT=${OUT:-runs/mixgrpo}
NUM_GPUS=${NUM_GPUS:-4}
ROLLOUT_TP=1
REWARD_TP=${REWARD_TP:-4}

capstan run verl-omni --out "$OUT" \
    trainer.use_v1=false \
    algorithm.adv_estimator=flow_grpo \
    data.train_files="$WORKSPACE/data/ocr/qwen_image/train.parquet" \
    data.val_files="$WORKSPACE/data/ocr/qwen_image/test.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=mix_grpo \
    actor_rollout_ref.model.path=${MODEL:-Qwen/Qwen-Image} \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.sample_strategy=random \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct} \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console"]' \
    trainer.project_name=capstan \
    trainer.experiment_name=qwen_image_ocr_lora_mixgrpo \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${STEPS:-12} \
    "$@"
