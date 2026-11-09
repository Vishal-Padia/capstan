import json
import os
import runpy
import sys

import capstan
from capstan.adapters import when_imported, wrap

TRAINER_MODULE = "verl_omni.trainer.main_diffusion"

PHASES = {
    "timing_s/step": "step",
    "timing_s/gen": "rollout",
    "timing_s/reward": "reward",
    "timing_s/old_log_prob": "recompute",
    "timing_s/update_actor": "update",
    "timing_s/update_weights": "sync",
}

# The drift hook runs before verl-omni logs the step, so log-probs wait here until Tracking.log gives the step number.
_pending = []


def _log(original):
    def log(self, data, step, *args, **kwargs):
        for arrays in _pending:
            capstan.record_logprobs(step=step, **arrays)
        _pending.clear()
        for key, phase in PHASES.items():
            if key in data:
                capstan.record_phase(phase, data[key], step)
        # Everything verl-omni logs, kept to cross-check against its own rollout_corr/* drift metrics.
        with open(capstan._out() / "verl_omni_metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **data}, default=float) + "\n")
        return original(self, data, step, *args, **kwargs)

    return log


# Called once per training step with the trainer recompute and the rollout log-probs of the same batch.
def _corr_metrics(original):
    def compute(log_prob, rollout_log_prob, timesteps=None):
        arrays = {"rollout": capstan._numpy(rollout_log_prob), "trainer": capstan._numpy(log_prob)}
        if timesteps is not None:
            arrays["timesteps"] = capstan._numpy(timesteps)
        _pending.append(arrays)
        return original(log_prob, rollout_log_prob, timesteps=timesteps)

    return compute


def install():
    if not os.environ.get("CAPSTAN_OUT"):
        return
    when_imported("verl.utils.tracking", lambda m: wrap(m.Tracking, "log", _log))
    when_imported(
        "verl_omni.trainer.diffusion.rollout_correction",
        lambda m: wrap(m, "compute_rollout_corr_metrics_from_logprobs", _corr_metrics),
    )


def _ray_init(original):
    # The trainer and its workers are Ray actors in other processes, so install() must run there too.
    def init(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        runtime_env["env_vars"] = {**env_vars, "CAPSTAN_OUT": os.environ["CAPSTAN_OUT"]}
        runtime_env["worker_process_setup_hook"] = "capstan.adapters.verl_omni.install"
        kwargs["runtime_env"] = runtime_env
        return original(*args, **kwargs)

    return init


def run(overrides):
    import ray

    install()
    wrap(ray, "init", _ray_init)
    sys.argv = [TRAINER_MODULE, *overrides]
    runpy.run_module(TRAINER_MODULE, run_name="__main__", alter_sys=True)
