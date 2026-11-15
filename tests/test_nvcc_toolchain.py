"""Compiler installation must fail closed and fingerprint all supporting inputs."""

import builtins
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "kernelbot_nvcc_toolchain", Path(__file__).parents[1] / "src/runners/nvcc_toolchain.py")
toolchain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(toolchain)


@pytest.fixture
def installation(tmp_path, monkeypatch):  # noqa: C901
    roots = {name: tmp_path / str(i) for i, name in enumerate(
        ("/nvcc-toolchain-input", "/usr/local/cuda", "/opt/kernelbot-pch"))}
    source = roots["/nvcc-toolchain-input"] / toolchain.RELEASE
    cuda, artifacts = roots["/usr/local/cuda"], roots["/opt/kernelbot-pch"]
    targets = {name: name if "/" in name else "bin/" + name for name in toolchain.BINARY_SHA256}
    hashes = {}
    files = {cuda / "bin/nvcc.profile": "CICC_PATH = $(TOP)/nvvm/bin\n"}
    for name, target in targets.items():
        files[source / name], files[cuda / target] = name, "original"
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    for name in ("bin/ptxas", "bin/nvlink", "bin/fatbinary", "nvvm/libdevice/libdevice.10.bc"):
        files[cuda / name] = name
    for name in ("fingerprint", "nvcc.py", "nvcc_common.py", "nvcc_warm.py"):
        files[artifacts / name] = name
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    original_import = builtins.__import__

    def sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pathlib" and globals and globals.get("__name__") == toolchain.__name__:
            return SimpleNamespace(Path=lambda path: roots.get(str(path), Path(path)))
        return original_import(name, globals, locals, fromlist, level)

    state = SimpleNamespace(source=source, cuda=cuda, artifacts=artifacts, targets=targets,
                            hashes=hashes, calls=[], version="V13.4.99",
                            backend="nvvm-next/bin/cicc",
                            packages="cuda-toolkit-13-4\t13.4.3-1\tinstall ok installed\n",
                            fail_arch=None)

    def output(command, **kwargs):
        state.calls.append(command)
        if command[0] == "dpkg-query":
            return state.packages
        return state.version if "--version" in command else state.backend

    def run(command, **kwargs):
        state.calls.append(command)
        if state.fail_arch and f"-arch=sm_{state.fail_arch}" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builtins, "__import__", sandbox_import)
    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", run)
    state.install = lambda: toolchain.install_nvcc_toolchain(toolchain.RELEASE, hashes)
    return state


def test_install_checks_every_architecture_and_is_reproducible(installation):
    s = installation
    s.install()
    for name, target in s.targets.items():
        assert (s.cuda / target).read_bytes() == (s.source / name).read_bytes()
        assert (s.cuda / target).stat().st_mode & 0o777 == 0o755
    compiled = [c[1] for c in s.calls if "-c" in c and "--dryrun" not in c]
    assert compiled == [f"-arch=sm_{a}" for a in ("75", "80", "89", "90a", "100", "100a")]
    before = (s.artifacts / "nvcc-fingerprint").read_text()
    manifest = json.loads((s.artifacts / "nvcc-toolchain.json").read_text())
    assert manifest["fingerprint"] == before.strip()
    s.install()
    assert (s.artifacts / "nvcc-fingerprint").read_text() == before


@pytest.mark.parametrize("failure", [*toolchain.BINARY_SHA256, "missing-hash", "extra-hash",
                                     "profile", "version", "backend", "smoke", "packages"])
def test_bad_installation_never_publishes_ready_fingerprint(installation, failure):
    s = installation
    if failure in s.hashes:
        (s.source / failure).write_bytes(b"corrupt")
    elif failure == "missing-hash":
        del s.hashes["nvcc"]
    elif failure == "extra-hash":
        s.hashes["unexpected"] = "0" * 64
    elif failure == "profile":
        (s.cuda / "bin/nvcc.profile").write_text("CICC_PATH=one\nCICC_PATH=two\n")
    else:
        field, value = {"version": ("version", "V13.3"), "backend": ("backend", "nvvm/bin/cicc"),
                        "smoke": ("fail_arch", "100a"), "packages": ("packages", "")}[failure]
        setattr(s, field, value)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        s.install()
    assert not (s.artifacts / "nvcc-fingerprint").exists()
    if failure in toolchain.BINARY_SHA256 or failure.endswith("hash"):
        assert not s.calls
        assert all((s.cuda / target).read_text() == "original" for target in s.targets.values())


@pytest.mark.parametrize("root,name", [
    ("artifacts", name) for name in ("fingerprint", "nvcc.py", "nvcc_common.py", "nvcc_warm.py")
] + [("cuda", name) for name in ("bin/ptxas", "bin/nvlink", "bin/fatbinary",
                               "nvvm/libdevice/libdevice.10.bc")] + [(None, "packages")])
def test_changed_supporting_input_invalidates_generation(installation, root, name):
    s = installation
    s.install()
    before = (s.artifacts / "nvcc-fingerprint").read_text()
    if root:
        path = getattr(s, root) / name
        path.write_bytes(path.read_bytes() + b" changed")
    else:
        s.packages = s.packages.replace("13.4.3-1", "13.4.3-2")
    s.install()
    assert (s.artifacts / "nvcc-fingerprint").read_text() != before
