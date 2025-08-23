import os

import pytest

REQUIRED = {
    "DISCORD_TOKEN": "dummy",
    "GITHUB_TOKEN": "dummy",
    "GITHUB_REPO": "dummy",
}

for k, v in REQUIRED.items():
    os.environ.setdefault(k, v)

@pytest.fixture(scope="session", autouse=True)
def _restore_env():
    old = {k: os.environ.get(k) for k in REQUIRED}
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
