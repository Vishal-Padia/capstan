import os
import sys

import numpy as np

from capstan.adapters import verl_omni, when_imported


def test_when_imported_patches_on_first_import(tmp_path, monkeypatch):
    (tmp_path / "capstan_fake_module.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    when_imported("capstan_fake_module", lambda m: seen.append(m.value))
    assert seen == []
    import capstan_fake_module  # noqa: F401

    assert seen == [1]
    del sys.modules["capstan_fake_module"]


def test_logprobs_are_written_with_the_step_verl_omni_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTAN_OUT", str(tmp_path))
    compute = verl_omni._corr_metrics(lambda log_prob, rollout_log_prob, timesteps=None: {})
    log = verl_omni._log(lambda self, data, step: None)

    compute(np.array([[0.1]]), np.array([[0.0]]))
    log(None, {"timing_s/gen": 1.5}, 7)

    saved = np.load(tmp_path / f"logprobs_step000007_pid{os.getpid()}.npz")
    assert saved["trainer"].tolist() == [[0.1]] and saved["rollout"].tolist() == [[0.0]]
