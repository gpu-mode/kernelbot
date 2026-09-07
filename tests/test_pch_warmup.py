"""The shared writer only accepts fixed profiles and publishes successful builds."""

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "src/runners"))
    module = importlib.import_module("modal_runner")
    events = []
    monkeypatch.setattr(
        module,
        "pch_volume",
        SimpleNamespace(
            reload=lambda: events.append("reload"), commit=lambda: events.append("commit")
        ),
    )
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: events.append(args))
    return module, events


def test_only_publishes_after_successful_warmup(runner):
    module, events = runner
    module.warm_pch.local("cuda-default,cuda-O3")
    assert events == [
        "reload",
        ["python3", "/opt/kernelbot-pch/warm.py", "cuda-default", "cuda-O3"],
        "commit",
    ]


@pytest.mark.parametrize(
    "profiles", ["-O3", "cuda-default; touch /tmp/oops", "cuda-unknown", "cpu-default,"]
)
def test_rejects_arbitrary_compiler_arguments(runner, profiles):
    module, events = runner
    with pytest.raises(ValueError, match="Unknown profiles"):
        module.warm_pch.local(profiles)
    assert not events


def test_failed_warmup_does_not_commit(runner, monkeypatch):
    module, events = runner

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.warm_pch.local("cuda-default")
    assert events == ["reload"]

