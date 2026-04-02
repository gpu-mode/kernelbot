import pytest

from libkernelbot import kernelguard
from libkernelbot.consts import SubmissionMode
from libkernelbot.utils import KernelBotError


def test_should_precheck_submission_enabled_modes(monkeypatch):
    monkeypatch.setenv("KERNELGUARD_ENABLED", "1")

    assert kernelguard.should_precheck_submission(SubmissionMode.BENCHMARK)
    assert kernelguard.should_precheck_submission(SubmissionMode.PROFILE)
    assert kernelguard.should_precheck_submission(SubmissionMode.LEADERBOARD)
    assert kernelguard.should_precheck_submission(SubmissionMode.PRIVATE)
    assert not kernelguard.should_precheck_submission(SubmissionMode.TEST)


def test_should_precheck_submission_disabled(monkeypatch):
    monkeypatch.delenv("KERNELGUARD_ENABLED", raising=False)
    assert not kernelguard.should_precheck_submission(SubmissionMode.BENCHMARK)


def test_enforce_submission_precheck_rejects_filtered_code(monkeypatch):
    monkeypatch.setenv("KERNELGUARD_ENABLED", "1")
    monkeypatch.setattr(
        kernelguard,
        "analyze_submission",
        lambda code: {
            "classification": "hacked",
            "should_filter": True,
            "filter_reason": "high_critical",
            "matched_patterns": [{"pattern": "MODULE_MUTATION"}],
        },
    )

    with pytest.raises(
        kernelguard.KernelGuardRejected,
        match=(
            r"Submission rejected by KernelGuard pre-check "
            r"\(high_critical\)\. Matched rules: MODULE_MUTATION\."
        ),
    ):
        kernelguard.enforce_submission_precheck("print('hello')", "submission.py")


def test_enforce_submission_precheck_fail_closed(monkeypatch):
    monkeypatch.setenv("KERNELGUARD_ENABLED", "1")
    monkeypatch.delenv("KERNELGUARD_FAIL_OPEN", raising=False)

    def _raise(_code):
        raise RuntimeError("boom")

    monkeypatch.setattr(kernelguard, "analyze_submission", _raise)

    with pytest.raises(
        KernelBotError,
        match="KernelGuard pre-check is unavailable right now. Please try again later.",
    ):
        kernelguard.enforce_submission_precheck("print('hello')", "submission.py")


def test_enforce_submission_precheck_fail_open(monkeypatch):
    monkeypatch.setenv("KERNELGUARD_ENABLED", "1")
    monkeypatch.setenv("KERNELGUARD_FAIL_OPEN", "1")

    def _raise(_code):
        raise RuntimeError("boom")

    monkeypatch.setattr(kernelguard, "analyze_submission", _raise)

    assert kernelguard.enforce_submission_precheck("print('hello')", "submission.py") is None


def test_analyze_submission_uses_cli_path(monkeypatch):
    expected = {"classification": "valid", "should_filter": False, "matched_patterns": []}
    monkeypatch.setattr(kernelguard, "_analyze_with_cli", lambda code: expected)

    assert kernelguard.analyze_submission("print('hello')") is expected
