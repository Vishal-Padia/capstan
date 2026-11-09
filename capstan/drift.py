import numpy as np

# Fixed before the first real run. 1e-4 is the recipe's own PPO clip window (verl-omni diffusion_loss.clip_ratio):
# drift that large moves the ratio as far as one policy update is allowed to.
RATIO_TOL = 1e-4
MAX_OVER_FRACTION = 0.01


def drift_stats(rollout_lp, trainer_lp, ratio_tol=RATIO_TOL):
    rollout_lp = np.asarray(rollout_lp, dtype=np.float64)
    trainer_lp = np.asarray(trainer_lp, dtype=np.float64)
    if rollout_lp.shape != trainer_lp.shape:
        raise ValueError(f"shape mismatch: rollout {rollout_lp.shape}, trainer {trainer_lp.shape}")
    # Inputs are verl-omni log-probs, already a mean over the latent, so d is per-element drift.
    d = trainer_lp - rollout_lp
    ratio = np.exp(d)
    over = (np.abs(ratio - 1) > ratio_tol).sum()
    return {
        "mean_abs": float(np.abs(d).mean()),
        "max_abs": float(np.abs(d).max()),
        "ratio_p50": float(np.percentile(ratio, 50)),
        "ratio_p99": float(np.percentile(ratio, 99)),
        "over_tol": int(over),
        "n": int(d.size),
        "passed": bool(over <= MAX_OVER_FRACTION * d.size),
    }
