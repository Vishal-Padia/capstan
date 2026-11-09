import logging
import os

import capstan
from capstan.adapters import when_imported, wrap

logger = logging.getLogger("capstan")

# The one name Capstan relies on: the pipeline verl-omni exports from this package (its __all__). Subclasses such as
# the MixGRPO pipeline call into it. The scheduler is found on the pipeline instance, not by import path.
PIPELINE_PACKAGE = "verl_omni.pipelines.qwen_image_flow_grpo"
PIPELINE_CLASS = "QwenImagePipelineWithLogProb"
TESTED_WITH = "verl-omni f50117e"

# (label, CUDA event) pairs for the request this process is generating; None outside a request.
_marks = None


def _mark(label):
    import torch

    event = torch.cuda.Event(enable_timing=True)
    event.record()
    _marks.append((label, event))


def _record(marks):
    labels = [label for label, _ in marks]
    if "diffuse_start" not in labels or "diffuse_end" not in labels:
        return
    marks[-1][1].synchronize()
    first, last = labels.index("diffuse_start"), labels.index("diffuse_end")
    events = [event for _, event in marks]
    capstan.record_phase("rollout/encode", events[0].elapsed_time(events[first]) / 1000)
    for before, (label, after) in zip(events[first : last - 1], marks[first + 1 : last], strict=True):
        capstan.record_phase(f"rollout/{label}_step", before.elapsed_time(after) / 1000)
    capstan.record_phase("rollout/decode", events[last].elapsed_time(events[-1]) / 1000)


def _forward(original):
    def forward(self, *args, **kwargs):
        global _marks
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            wrap(type(scheduler), "step", _step)
        _marks = []
        _mark("start")
        try:
            return original(self, *args, **kwargs)
        finally:
            _mark("end")
            marks, _marks = _marks, None
            _record(marks)

    return forward


def _diffuse(original):
    def diffuse(self, *args, **kwargs):
        if _marks is not None:
            _mark("diffuse_start")
        result = original(self, *args, **kwargs)
        if _marks is not None:
            _mark("diffuse_end")
        return result

    return diffuse


# ODE and SDE steps run the same transformer forward; only the scheduler step adds noise and a log-prob.
def _step(original):
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _marks is not None:
            level = kwargs.get("noise_level", 0.0)
            noisy = bool((level > 0).any()) if hasattr(level, "any") else level > 0
            _mark("sde" if noisy else "ode")
        return result

    return step


def _patch_pipeline(package):
    pipeline = getattr(package, PIPELINE_CLASS, None)
    if pipeline is None or not hasattr(pipeline, "diffuse"):
        # A profiler must not break training: record nothing, say why, and let capstan report flag the gap.
        message = "%s.%s.diffuse not found; rollout split not recorded (tested with %s)"
        logger.warning(message, PIPELINE_PACKAGE, PIPELINE_CLASS, TESTED_WITH)
        return
    wrap(pipeline, "forward", _forward)
    wrap(pipeline, "diffuse", _diffuse)


# vLLM general plugin: vllm-omni loads it in every engine process, including the diffusion worker.
def register():
    if not os.environ.get("CAPSTAN_OUT"):
        return
    when_imported(PIPELINE_PACKAGE, _patch_pipeline)
