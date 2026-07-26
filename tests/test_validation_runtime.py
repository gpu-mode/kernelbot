import json

from libkernelbot.validation_runtime import run_validation_config


def test_validation_runtime_executes_problem_owned_entrypoint():
    config = {
        "name": "optimizer",
        "version": "optimizer-v1",
        "main": "validation.py",
        "sources": {
            "validation.py": (
                "import json, os\n"
                "config = json.loads(os.environ['KERNELBOT_VALIDATION_CONFIG'])\n"
                "print(json.dumps({'contract_version': config['version'], "
                "'passed_shapes': 1, 'total_shapes': 1}))\n"
            ),
            "submission.py": "def custom_kernel(value): return value\n",
        },
        "shapes": [{"n": 32}],
        "settings": {},
        "timeout": 10,
    }

    result = run_validation_config(config)

    assert result == {
        "status": "completed",
        "result": {
            "contract_version": "optimizer-v1",
            "passed_shapes": 1,
            "total_shapes": 1,
        },
    }


def test_validation_runtime_does_not_return_untrusted_output():
    result = run_validation_config(
        {
            "name": "optimizer",
            "version": "optimizer-v1",
            "main": "validation.py",
            "sources": {
                "validation.py": "print('submission secret')\n",
                "submission.py": "SECRET = 'must not leak'\n",
            },
            "timeout": 10,
        }
    )

    assert result["status"] == "failed"
    assert "submission secret" not in json.dumps(result)
    assert "must not leak" not in json.dumps(result)


def test_validation_runtime_rejects_path_traversal():
    result = run_validation_config(
        {
            "name": "optimizer",
            "version": "optimizer-v1",
            "main": "validation.py",
            "sources": {
                "validation.py": "print('{}')\n",
                "../submission.py": "SECRET = True\n",
            },
            "timeout": 10,
        }
    )

    assert result["status"] == "failed"
    assert "escapes workspace" in result["error"]
