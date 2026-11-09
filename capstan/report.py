import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from capstan.drift import MAX_OVER_FRACTION, RATIO_TOL, drift_stats

TRAINER_PHASES = ["rollout", "reward", "recompute", "update", "sync"]
ROLLOUT_PARTS = {
    "encode": "Text encode",
    "ode_step": "ODE steps",
    "sde_step": "SDE-window steps",
    "decode": "VAE decode",
}


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def profile(run, skip):
    rows = [json.loads(line) for f in sorted(run.glob("phases_*.jsonl")) for line in f.read_text().splitlines()]
    steps = {r["step"]: r for r in rows if r["phase"] == "step" and r["step"] > skip}
    if not steps:
        raise SystemExit(f"no steps after skipping {skip} in {run}")
    n = len(steps)
    total = sum(r["seconds"] for r in steps.values()) / n
    per_step = {
        p: sum(r["seconds"] for r in rows if r["phase"] == p and r["step"] in steps) / n for p in TRAINER_PHASES
    }
    per_step["everything else, idle included"] = total - sum(per_step.values())
    trainer_rows = [[p, f"{s:.2f}", f"{s / total:.1%}"] for p, s in per_step.items()]
    trainer_rows.append(["step", f"{total:.2f}", "100%"])

    # Rollout workers do not know the training step, so their records go to the step whose time window holds them.
    windows = [(r["end"] - r["seconds"], r["end"]) for r in steps.values()]
    parts = defaultdict(list)
    for r in rows:
        name = r["phase"].removeprefix("rollout/")
        if name in ROLLOUT_PARTS and any(start <= r["end"] <= end for start, end in windows):
            parts[name].append(r["seconds"])
    worker_total = sum(sum(v) for v in parts.values())
    part_rows = [
        [label, f"{sum(parts[k]) / n:.2f}", f"{sum(parts[k]) / worker_total:.1%}", len(parts[k]) // n]
        for k, label in ROLLOUT_PARTS.items()
        if parts[k]
    ]

    out = [f"# Profile\n\nMean over {n} training steps, skipping the first {skip} as warm-up.\n"]
    out.append(_table(["Phase", "Time per step (s)", "Share of step"], trainer_rows))
    out.append("\n## Inside rollout\n")
    if part_rows:
        out.append("GPU time summed over rollout workers, per training step.\n")
        out.append(_table(["Part", "GPU seconds per step", "Share of rollout GPU time", "Calls per step"], part_rows))
    else:
        out.append(
            "Not recorded. The split needs the vLLM plugin to find verl-omni's pipeline; check the rollout logs."
        )
    return "\n".join(out) + "\n"


def drift(run):
    by_step = defaultdict(list)
    for f in run.glob("logprobs_step*_pid*.npz"):
        by_step[int(re.search(r"step(\d+)", f.name)[1])].append(np.load(f))
    if not by_step:
        raise SystemExit(f"no log-prob files in {run}")

    def stats(arrays):
        return drift_stats(
            np.concatenate([a["rollout"] for a in arrays]), np.concatenate([a["trainer"] for a in arrays])
        )

    def row(label, s):
        ratios = f"{s['ratio_p50']:.6f}", f"{s['ratio_p99']:.6f}"
        return [label, s["n"], f"{s['mean_abs']:.3g}", f"{s['max_abs']:.3g}", *ratios, s["over_tol"]]

    rows = [row(step, stats(by_step[step])) for step in sorted(by_step)]
    overall = stats([a for arrays in by_step.values() for a in arrays])
    rows.append(row("all", overall))
    verdict = "passed" if overall["passed"] else "failed"
    out = [
        "# Drift\n",
        f"Threshold, fixed before the first real run: |ratio − 1| ≤ {RATIO_TOL:g} for at least "
        f"{1 - MAX_OVER_FRACTION:.0%} of (sample, SDE step) entries. This run {verdict}: "
        f"{overall['over_tol']} of {overall['n']} entries over.\n",
        _table(
            ["Step", "Entries", "Mean abs drift", "Max abs drift", "Ratio p50", "Ratio p99", "Over threshold"], rows
        ),
    ]
    return "\n".join(out) + "\n"


def write(run, out, skip):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.md").write_text(profile(run, skip))
    (out / "drift.md").write_text(drift(run))
