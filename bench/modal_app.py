import os
import re
import subprocess
import time
from pathlib import Path

import modal

VERL_OMNI_COMMIT = "81f3b65a43e5ee13f499c1cc2718004a372c29ff"
REPO = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system vllm==0.33.0 --torch-backend=cu130",
        "git clone https://github.com/verl-project/verl-omni.git /verl-omni"
        f" && cd /verl-omni && git checkout {VERL_OMNI_COMMIT}",
        "cd /verl-omni && uv pip install --system -e '.[gpu,ocr]' --torch-backend=cu130",
        "hf download tiny-random/Qwen-Image --local-dir /models/tiny-random/Qwen-Image",
        "hf download tiny-random/qwen3-vl --local-dir /models/tiny-random/qwen3-vl",
    )
    .add_local_dir(
        REPO,
        "/capstan",
        copy=True,
        ignore=[".venv", ".git", "**/__pycache__", "runs", "capstan.md", "plan.md", "drafts"],
    )
    .run_commands("uv pip install --system -e /capstan")
)
app = modal.App("capstan", image=image)
volume = modal.Volume.from_name("capstan-models", create_if_missing=True)

# Shrinks the recipe to fit the tiny random-weight models on one small GPU; checks recording, measures nothing.
TINY = [
    "actor_rollout_ref.model.attn_backend=native",
    "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA",
    "actor_rollout_ref.rollout.enforce_eager=True",
    "actor_rollout_ref.model.pipeline.height=256",
    "actor_rollout_ref.model.pipeline.width=256",
    "actor_rollout_ref.rollout.pipeline.height=256",
    "actor_rollout_ref.rollout.pipeline.width=256",
    "actor_rollout_ref.model.tokenizer_path=/models/tiny-random/Qwen-Image/tokenizer",
    "actor_rollout_ref.model.lora_rank=8",
    "actor_rollout_ref.model.lora_alpha=16",
    "actor_rollout_ref.model.target_modules=all-linear",
    "data.train_batch_size=2",
    "actor_rollout_ref.rollout.n=4",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.layered_summon=False",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
    "actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0",
    "reward.reward_model.rollout.gpu_memory_utilization=0.3",
]
STEP_MODE = ["actor_rollout_ref.rollout.step_execution=True", "actor_rollout_ref.rollout.max_num_seqs=16"]


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(msg):
    print(f"[capstan {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Writes the full output to log_path and prints only progress lines, plus a heartbeat with the latest output every
# two minutes, so the Modal dashboard shows where a run is without the full training log.
def stream(cmd, log_path, on_step=None, **kwargs):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    full = open(log_path, "a", buffering=1)
    tail, beat = [], time.time()
    for line in proc.stdout:
        full.write(line)
        tail = (tail + [line])[-200:]
        if line.strip() and time.time() - beat > 120:
            log("running; latest: " + ANSI.sub("", line).strip()[:200])
            beat = time.time()
        if m := re.search(r"\bstep:(\d+) - ", line):
            t = re.search(r"timing_s/step:([\d.]+)", line)
            log(f"training step {m[1]} done" + (f" in {float(t[1]):.0f}s" if t else ""))
            if on_step:
                on_step()
        elif "Traceback" in line or "Error" in line.split(":")[0] or "rollout split not recorded" in line:
            log(line.rstrip()[:300])
    full.close()
    if proc.wait():
        print("".join(tail), flush=True)
        raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}")


def run_recipe(out, env, overrides, on_step=None):
    cmd = ["bash", "/capstan/bench/run_mixgrpo.sh", *overrides]
    stream(cmd, f"{out}/train.log", on_step=on_step, cwd="/verl-omni", env={**os.environ, "OUT": out, **env})
    log("training finished; writing report")
    subprocess.run(["capstan", "report", out, "--skip", env.get("SKIP", "2")], check=True)
    report = (Path(out) / "profile.md").read_text() + "\n" + (Path(out) / "drift.md").read_text()
    print(report, flush=True)
    return report


@app.function(volumes={"/vol": volume}, timeout=3600, cpu=4)
def stage():
    sh = lambda cmd: subprocess.run(["bash", "-c", cmd], check=True, cwd="/verl-omni")  # noqa: E731
    for name in ("Qwen/Qwen-Image", "Qwen/Qwen3-VL-8B-Instruct"):
        log(f"downloading {name}")
        sh(f"hf download {name} --local-dir /vol/{name.split('/')[1]}")
    log("building the OCR dataset")
    sh(
        "git clone -q --depth 1 --filter=blob:none --sparse https://github.com/yifan123/flow_grpo /tmp/fg"
        " && git -C /tmp/fg sparse-checkout set dataset/ocr"
    )
    sh(
        "python examples/flowgrpo_trainer/data_process/qwenimage_ocr.py"
        " --input_dir /tmp/fg/dataset/ocr --output_dir /vol/data/ocr/qwen_image"
    )
    volume.commit()
    log("staged")


@app.function(gpu="A10G", volumes={"/vol": volume}, timeout=2400)
def dry(name: str, step_mode: bool = False):
    log(f"tiny MixGRPO dry run, {'step' if step_mode else 'request'} mode, 3 steps")
    env = {
        "WORKSPACE": "/vol",
        "NUM_GPUS": "1",
        "REWARD_TP": "1",
        "STEPS": "3",
        "SKIP": "0",
        "MODEL": "/models/tiny-random/Qwen-Image",
        "REWARD_MODEL": "/models/tiny-random/qwen3-vl",
    }
    try:
        return run_recipe(f"/vol/runs/{name}", env, TINY + (STEP_MODE if step_mode else []))
    finally:
        volume.commit()


@app.function(gpu="H100", volumes={"/vol": volume}, timeout=3600)
def batch_invariance(name: str, a: int = 1, b: int = 16):
    outs = []
    for batch in (a, b):
        log(f"generating at batch {batch}")
        out = f"/vol/runs/{name}/batch{batch}.npz"
        stream(
            [
                "python",
                "/capstan/bench/batch_invariance.py",
                "generate",
                "--model",
                "/vol/Qwen-Image",
                "--batch",
                str(batch),
                "--out",
                out,
            ],
            f"/vol/runs/{name}/batch{batch}.log",
            cwd="/verl-omni",
        )
        outs.append(out)
    table = subprocess.run(
        ["python", "/capstan/bench/batch_invariance.py", "compare", *outs], check=True, capture_output=True, text=True
    ).stdout
    print(table, flush=True)
    Path(f"/vol/runs/{name}/batch_invariance.md").write_text(table)
    volume.commit()
    return table


@app.function(gpu="H100:4", memory=262144, volumes={"/vol": volume}, timeout=3 * 3600)
def mixgrpo(name: str, steps: int = 5, step_mode: bool = False):
    out = f"/vol/runs/{name}"
    log(f"Qwen-Image MixGRPO on 4x H100, {steps} steps, {'step' if step_mode else 'request'} mode -> {out}")
    env = {
        "WORKSPACE": "/vol",
        "STEPS": str(steps),
        "MODEL": "/vol/Qwen-Image",
        "REWARD_MODEL": "/vol/Qwen3-VL-8B-Instruct",
    }
    try:
        # Committing after every step means a timeout or crash still leaves the finished steps on the volume.
        return run_recipe(out, env, STEP_MODE if step_mode else [], on_step=volume.commit)
    finally:
        volume.commit()


def download(name):
    dest = REPO / "logs" / name
    for entry in volume.listdir(f"runs/{name}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        path = dest / Path(entry.path).relative_to(f"runs/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
    print(f"saved {dest}")


# Runs a job and saves everything it recorded (full log, timings, log-probs, reports) to logs/<name>/ locally.
@app.local_entrypoint()
def run(job: str, name: str = "", steps: int = 5, step_mode: bool = False):
    name = name or f"{job}-{time.strftime('%Y%m%d-%H%M%S')}"
    jobs = {
        "dry": lambda: dry.remote(name, step_mode),
        "batch-invariance": lambda: batch_invariance.remote(name),
        "mixgrpo": lambda: mixgrpo.remote(name, steps, step_mode),
    }
    try:
        jobs[job]()
    finally:
        download(name)


# For runs started with --detach, or whose local client went away before the download.
@app.local_entrypoint()
def fetch(name: str):
    download(name)
