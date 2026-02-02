import copy
import json

import pytest

from libkernelbot.consts import SubmissionMode
from libkernelbot.task import (
    CudaTaskData,
    Language,
    LeaderboardDefinition,
    LeaderboardTask,
    PythonTaskData,
    RankCriterion,
    build_task_config,
    make_task_definition,
)
from libkernelbot.utils import KernelBotError


@pytest.fixture()
def leaderboard_task():
    return LeaderboardTask(
        lang=Language.Python,
        files={"test.py": "code", "main.py": "@SUBMISSION@"},
        config=PythonTaskData(main="main.py"),
        ranking_by=RankCriterion.GEOM,
        test_timeout=120,
        tests=[{"input_size": 1000, "dtype": "float32"}, {"input_size": 5000, "dtype": "float16"}],
        benchmarks=[
            {"input_size": 10000, "dtype": "float32"},
            {"input_size": 50000, "dtype": "float16"},
        ],
    )


def test_from_dict_python_task():
    """Test creating LeaderboardTask from dict with Python config"""
    data = {
        "lang": "py",
        "files": {"main.py": "print('hello')"},
        "config": {"main": "main"},
        "tests": [{"input": "test"}],
        "ranking_by": "last",
    }

    old_value = copy.deepcopy(data)
    task = LeaderboardTask.from_dict(data)

    assert task.lang.value == "py"
    assert task.files == {"main.py": "print('hello')"}
    assert isinstance(task.config, PythonTaskData)
    assert task.ranking_by == RankCriterion.LAST
    assert task.tests == [{"input": "test"}]

    # check that from_dict does not modify data
    assert data == old_value


def test_from_dict_cuda_task():
    """Test creating LeaderboardTask from dict with CUDA config"""
    """Test creating LeaderboardTask from dict with CUDA config"""
    data = {
        "lang": "cu",
        "files": {"kernel.cu": "__global__ void test(){}"},
        "config": {
            "sources": ["kernel.cu"],
            "include_dirs": ["/usr/include"],
            "defines": {"DEBUG": "1"},
            "compile_flags": ["-O2"],
        },
    }

    old_value = copy.deepcopy(data)
    task = LeaderboardTask.from_dict(data)

    assert task.lang == Language.CUDA
    assert isinstance(task.config, CudaTaskData)
    assert task.config.sources == ["kernel.cu"]
    assert task.config.include_dirs == ["/usr/include"]
    assert task.config.defines == {"DEBUG": "1"}
    assert task.config.compile_flags == ["-O2"]
    assert task.ranking_by == RankCriterion.LAST

    # check that from_dict does not modify data
    assert data == old_value


def test_type_mismatch():
    """Claim CUDA but supply python"""
    with pytest.raises(TypeError):
        _ = LeaderboardTask(
            lang=Language.CUDA, files={"test.py": "code"}, config=PythonTaskData(main="main")
        )


def test_to_dict(leaderboard_task):
    """Test converting LeaderboardTask to dict"""
    result = leaderboard_task.to_dict()

    assert result["lang"] == Language.Python.value
    assert result["files"] == {
        "test.py": "code",
        "main.py": "@SUBMISSION@",
    }
    assert result["ranking_by"] == RankCriterion.GEOM.value
    assert result["test_timeout"] == 120
    assert result["tests"] == [
        {"input_size": 1000, "dtype": "float32"},
        {"input_size": 5000, "dtype": "float16"},
    ]
    assert result["benchmarks"] == [
        {"input_size": 10000, "dtype": "float32"},
        {"input_size": 50000, "dtype": "float16"},
    ]


def test_serialization_roundtrip(leaderboard_task):
    """Test to_str and from_str work together"""
    json_str = leaderboard_task.to_str()
    reconstructed = LeaderboardTask.from_str(json_str)

    assert reconstructed == leaderboard_task


def test_build_task_config_python(leaderboard_task):
    """Test build_task_config with Python task and submission content."""
    submission_content = "print('Hello World')"
    arch = "sm_80"
    mode = SubmissionMode.PRIVATE

    result = build_task_config(
        task=leaderboard_task, submission_content=submission_content, arch=arch, mode=mode
    )

    # make sure result is serializable
    json.dumps(result)

    expected = {
        "main": "main.py",
        "sources": {"test.py": "code", "main.py": "print('Hello World')"},
        "lang": "py",
        "arch": "sm_80",
        "benchmarks": [
            {"input_size": 10000, "dtype": "float32"},
            {"input_size": 50000, "dtype": "float16"},
        ],
        "tests": [
            {"input_size": 1000, "dtype": "float32"},
            {"input_size": 5000, "dtype": "float16"},
        ],
        "mode": mode.value,
        "multi_gpu": False,
        "test_timeout": 120,
        "benchmark_timeout": 180,
        "ranked_timeout": 180,
        "ranking_by": "geom",
        "seed": None,
    }

    assert result == expected


def test_build_task_config_cuda():
    """Test build_task_config with CUDA task and submission content."""
    submission_content = "print('Hello World')"
    arch = "sm_80"
    mode = SubmissionMode.PRIVATE
    task = LeaderboardTask(
        lang=Language.CUDA,
        files={"test.cu": "code", "submission.cu": "@SUBMISSION@", "test.cuh": "header"},
        config=CudaTaskData(
            sources=["test.cu", "submission.cu"],
            include_dirs=["/usr/include"],
            defines={"DEBUG": "1"},
        ),
        ranking_by=RankCriterion.GEOM,
        test_timeout=120,
        tests=[{"input_size": 1000, "dtype": "float32"}, {"input_size": 5000, "dtype": "float16"}],
        benchmarks=[
            {"input_size": 10000, "dtype": "float32"},
            {"input_size": 50000, "dtype": "float16"},
        ],
    )
    result = build_task_config(
        task=task, submission_content=submission_content, arch=arch, mode=mode
    )

    # make sure result is serializable
    json.dumps(result)

    expected = {
        "sources": {"submission.cu": "print('Hello World')", "test.cu": "code"},
        "headers": {"test.cuh": "header"},
        "include_dirs": ["/usr/include"],
        "lang": "cu",
        "arch": "sm_80",
        "benchmarks": [
            {"input_size": 10000, "dtype": "float32"},
            {"input_size": 50000, "dtype": "float16"},
        ],
        "tests": [
            {"input_size": 1000, "dtype": "float32"},
            {"input_size": 5000, "dtype": "float16"},
        ],
        "mode": mode.value,
        "multi_gpu": False,
        "test_timeout": 120,
        "benchmark_timeout": 180,
        "ranked_timeout": 180,
        "ranking_by": "geom",
        "seed": None,
        "compile_flags": [],
        "defines": {"DEBUG": "1"},
    }

    assert result == expected


def test_make_task_definition(task_directory):
    """Test make_task_definition with a complete YAML structure"""

    # Test the function
    result = make_task_definition(task_directory / "task.yml")

    # Verify LeaderboardDefinition structure
    assert isinstance(result, LeaderboardDefinition)
    assert result.description == "Test task description"
    assert result.templates == {"Python": "# Python template", "CUDA": "// CUDA template"}

    # Verify the task
    task = result.task
    assert task.lang == Language.Python
    assert task.files == {"kernel.py": "def kernel(): pass", "submission.py": "@SUBMISSION@"}
    assert task.ranking_by == RankCriterion.GEOM
    assert task.test_timeout == 120
    assert task.tests == [{"input_size": 1000, "dtype": "float32"}]
    assert task.benchmarks == [{"input_size": 10000, "dtype": "float32"}]
    assert isinstance(task.config, PythonTaskData)
    assert task.config.main == "kernel.py"


def test_multi_gpu_task(task_directory):
    """Test make_task_definition with a multi-GPU task"""
    orig = (task_directory / "task.yml").read_text()
    (task_directory / "task.yml").write_text(orig + "\nmulti_gpu: true")

    # no world size specified => Error
    with pytest.raises(KernelBotError, match="does not specify world_size"):
        make_task_definition(task_directory / "task.yml")

    result = make_task_definition(task_directory / "multi-task.yml")
    assert result.task.multi_gpu is True
