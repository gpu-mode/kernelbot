"""CPU regression checks for reproducibility and lossless result export."""

import hashlib
import json
import random
from pathlib import Path

import pytest
from export_result import emit_result
from launch import decode_result, prepare_payload


def test_large_evidence_roundtrip_and_checksum(capsys):
    """Outputs larger than a provider log line must survive multiple chunks."""
    result = {"log": random.Random(42).randbytes(150000).hex(), "passed": False}
    emit_result(result)
    lines = capsys.readouterr().out.splitlines()
    chunks = [line.split("=", 1)[1] for line in lines if line.startswith("EXPERIMENT_CHUNK=")]
    checksum = lines[-1].split("=", 1)[1]
    assert len(chunks) > 1
    assert max(map(len, lines)) < 25000
    assert decode_result(chunks, checksum) == result
    with pytest.raises(ValueError, match="checksum"):
        decode_result(chunks, "0" * 64)


def test_payload_preserves_evaluator_and_submission(tmp_path):
    """Build from current repository sources and keep the stated benchmark shapes."""
    repo = Path(__file__).resolve().parents[2]
    hashes = prepare_payload(tmp_path, repo)
    for name in ("run_eval.py", "consts.py"):
        source = (repo / "src/libkernelbot" / name).read_bytes()
        assert (tmp_path / "source/libkernelbot" / name).read_bytes() == source
        assert hashes[f"source/libkernelbot/{name}"] == hashlib.sha256(source).hexdigest()
    for workload, filename in [("inline", "submission_cuda_inline.py"), ("triton", "submission_triton.py")]:
        config = json.loads((tmp_path / f"{workload}.json").read_text())
        assert config["sources"]["submission.py"] == (repo / "examples/vectoradd_py" / filename).read_text()
        assert config["sources"]["eval.py"] == (repo / "examples/eval.py").read_text()
        assert [case["size"] for case in config["benchmarks"]] == [1024, 2048, 4096]
        assert config["mode"] == "benchmark" and not config["multi_gpu"]
