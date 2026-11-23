from pathlib import Path

from libkernelbot.run_eval import SystemInfo, run_pytorch_script

EVAL_WRITES_NEXT_TO_ITSELF = """
import os
from pathlib import Path

Path("kernel.mojo").write_text("previous user's kernel")
Path("__mojocache__").mkdir()
with os.fdopen(int(os.environ["POPCORN_FD"]), "w") as out:
    out.write(f"check: pass\\ncwd: {os.getcwd()}\\n")
"""


def test_python_submission_runs_in_private_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run_pytorch_script(
        sources={"eval.py": EVAL_WRITES_NEXT_TO_ITSELF, "submission.py": ""},
        main="eval.py",
        mode="test",
        system=SystemInfo(),
        tests="size: 1\n",
    )

    assert result.run.passed, result.run.stderr
    run_dir = Path(result.run.result["cwd"])
    assert run_dir.resolve() != tmp_path.resolve()
    assert not run_dir.exists()
    assert list(tmp_path.iterdir()) == []
    assert Path.cwd() == tmp_path
