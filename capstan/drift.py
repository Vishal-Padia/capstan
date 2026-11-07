import numpy as np


def drift_stats(rollout_lp, trainer_lp, ratio_tol):
    rollout_lp = np.asarray(rollout_lp, dtype=np.float64)
    trainer_lp = np.asarray(trainer_lp, dtype=np.float64)
    if rollout_lp.shape != trainer_lp.shape:
        raise ValueError(f"shape mismatch: rollout {rollout_lp.shape}, trainer {trainer_lp.shape}")
    # Inputs are verl-omni log-probs, already a mean over the latent, so d is per-element drift.
    d = trainer_lp - rollout_lp
    ratio = np.exp(d)
    return {
        "mean_abs": float(np.abs(d).mean()),
        "max_abs": float(np.abs(d).max()),
        "ratio_p50": float(np.percentile(ratio, 50)),
        "ratio_p99": float(np.percentile(ratio, 99)),
        "over_tol": int((np.abs(ratio - 1) > ratio_tol).sum()),
        "n": int(d.size),
    }
