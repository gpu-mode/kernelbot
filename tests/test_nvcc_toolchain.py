"""Private compiler installation and cache-generation release gates, without CUDA."""

import builtins
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "kernelbot_nvcc_toolchain", Path(__file__).parents[1] / "src/runners/nvcc_toolchain.py"
)
toolchain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(toolchain)


@pytest.fixture
def installation(tmp_path, monkeypatch):  # noqa: C901
    roots = {
        "/nvcc-toolchain-input": tmp_path / "input",
        "/usr/local/cuda": tmp_path / "cuda",
        "/opt/kernelbot-pch": tmp_path / "pch",
    }
    source = roots["/nvcc-toolchain-input"] / toolchain.RELEASE
    cuda, artifacts = roots["/usr/local/cuda"], roots["/opt/kernelbot-pch"]
    targets = {
        "nvcc": "bin/nvcc",
        "cudafe++": "bin/cudafe++",
        "nvvm/bin/cicc": "nvvm/bin/cicc",
        "nvvm-next/bin/cicc": "nvvm-next/bin/cicc",
    }

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    hashes = {}
    for name, target in targets.items():
        data = f"pinned private compiler: {name}".encode()
        write(source / name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        write(cuda / target, f"original toolkit: {name}".encode())
    write(cuda / "bin/nvcc.profile", b"CICC_PATH = $(TOP)/nvvm/bin\nOTHER = keep\n")
    for name in ("bin/ptxas", "bin/nvlink", "bin/fatbinary", "nvvm/libdevice/libdevice.10.bc"):
        write(cuda / name, name.encode())
    for name in ("fingerprint", "nvcc.py", "nvcc_common.py", "nvcc_warm.py"):
        write(artifacts / name, name.encode())

    # The image-build function imports pathlib internally. Redirect only that
    # import; pytest and the standard library keep their ordinary Path objects.
    original_import = builtins.__import__

    def sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pathlib" and globals and globals.get("__name__") == toolchain.__name__:
            return SimpleNamespace(Path=lambda path: roots.get(str(path), Path(path)))
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", sandbox_import)
    state = SimpleNamespace(
        source=source,
        cuda=cuda,
        artifacts=artifacts,
        targets=targets,
        hashes=hashes,
        calls=[],
        version="Cuda compilation tools, release 13.4, V13.4.99\n",
        backend="nvvm-next/bin/cicc",
        packages=(
            "cuda-toolkit-13-4\t13.4.3-1\tinstall ok installed\n"
            "cuda-nvcc-13-4\t13.4.92-1\tinstall ok installed\n"
            "libcublas-13-4\t13.4.0.1-1\tinstall ok installed\n"
            "removed-13-4\t1.0-1\tdeinstall ok config-files\n"
        ),
        fail_arch=None,
    )

    def output(command, **kwargs):
        state.calls.append(command)
        if command[0] == "dpkg-query":
            return state.packages
        return state.version if "--version" in command else state.backend

    def run(command, **kwargs):
        state.calls.append(command)
        if state.fail_arch and f"-arch=sm_{state.fail_arch}" in command:
            raise subprocess.CalledProcessError(1, command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled object")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", run)
    state.install = lambda: toolchain.install_nvcc_toolchain(toolchain.RELEASE, hashes)
    return state


def test_installs_verified_compilers_and_publishes_reproducible_generation(installation):
    state = installation
    state.install()
    for name, target in state.targets.items():
        installed = state.cuda / target
        assert installed.read_bytes() == (state.source / name).read_bytes()
        assert installed.stat().st_mode & 0o777 == 0o755
    profile = (state.cuda / "bin/nvcc.profile").read_text()
    assert "CICC_PATH        = $(TOP)/nvvm-next/bin" in profile
    assert "OTHER = keep" in profile

    manifest = json.loads((state.artifacts / "nvcc-toolchain.json").read_text())
    generation = manifest.pop("fingerprint")
    assert manifest["smoke_architectures"] == ["75", "80", "89", "90a", "100", "100a"]
    compiled = [call[1] for call in state.calls if "-c" in call and "--dryrun" not in call]
    assert compiled == [f"-arch=sm_{arch}" for arch in manifest["smoke_architectures"]]
    assert manifest["headers_fingerprint"] == "fingerprint"
    assert manifest["cuda_packages"] == {
        "cuda-toolkit-13-4": "13.4.3-1",
        "cuda-nvcc-13-4": "13.4.92-1",
        "libcublas-13-4": "13.4.0.1-1",
    }
    assert generation == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert (state.artifacts / "nvcc-fingerprint").read_text().strip() == generation
    state.install()
    assert (state.artifacts / "nvcc-fingerprint").read_text().strip() == generation


@pytest.mark.parametrize("damaged", ["nvcc", "cudafe++", "nvvm/bin/cicc", "nvvm-next/bin/cicc"])
def test_checksum_failure_changes_no_installed_binary(installation, damaged):
    state = installation
    original = {target: (state.cuda / target).read_bytes() for target in state.targets.values()}
    (state.source / damaged).write_bytes(b"unverified compiler")
    with pytest.raises(ValueError, match="checksum mismatch"):
        state.install()
    assert original == {
        target: (state.cuda / target).read_bytes() for target in state.targets.values()
    }
    assert not state.calls
    assert not (state.artifacts / "nvcc-fingerprint").exists()


@pytest.mark.parametrize(
    "missing", ["nvcc", "cudafe++", "nvvm/bin/cicc", "nvvm-next/bin/cicc", None]
)
def test_checksum_manifest_must_cover_exactly_installed_binaries(installation, missing):
    state = installation
    original = {target: (state.cuda / target).read_bytes() for target in state.targets.values()}
    if missing:
        del state.hashes[missing]
    else:
        state.hashes["unrecognized-compiler"] = "a" * 64
    with pytest.raises(ValueError, match="exactly the four NVIDIA compiler binaries"):
        state.install()
    assert original == {
        target: (state.cuda / target).read_bytes() for target in state.targets.values()
    }
    assert not state.calls
    assert not (state.artifacts / "nvcc-fingerprint").exists()


@pytest.mark.parametrize("profile", ["OTHER = keep\n", "CICC_PATH = one\nCICC_PATH = two\n"])
def test_ambiguous_backend_profile_blocks_release(installation, profile):
    state = installation
    (state.cuda / "bin/nvcc.profile").write_text(profile)
    with pytest.raises(ValueError, match="exactly one CICC_PATH"):
        state.install()
    assert not state.calls
    assert not (state.artifacts / "nvcc-fingerprint").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "version",
            "Cuda compilation tools, release 13.4, V13.4.92\n",
            "Unexpected NVIDIA compiler",
        ),
        ("backend", "nvvm/bin/cicc", "Wrong CUDA compiler backend"),
    ],
)
def test_version_and_backend_mismatch_block_release(installation, field, value, message):
    state = installation
    setattr(state, field, value)
    with pytest.raises(ValueError, match=message):
        state.install()
    assert not (state.artifacts / "nvcc-toolchain.json").exists()
    assert not (state.artifacts / "nvcc-fingerprint").exists()


def test_failed_architecture_smoke_blocks_release(installation):
    state = installation
    state.fail_arch = "100a"
    with pytest.raises(subprocess.CalledProcessError):
        state.install()
    assert not (state.artifacts / "nvcc-toolchain.json").exists()
    assert not (state.artifacts / "nvcc-fingerprint").exists()


def test_supporting_library_upgrade_invalidates_cache_generation(installation):
    state = installation
    state.install()
    before = (state.artifacts / "nvcc-fingerprint").read_text()
    state.packages = state.packages.replace(
        "libcublas-13-4\t13.4.0.1-1", "libcublas-13-4\t13.4.0.2-1"
    )
    state.install()
    assert (state.artifacts / "nvcc-fingerprint").read_text() != before


@pytest.mark.parametrize("packages", ["", "removed-13-4\t1.0-1\tdeinstall ok config-files\n"])
def test_missing_installed_package_inventory_blocks_release(installation, packages):
    state = installation
    state.packages = packages
    with pytest.raises(ValueError, match="No installed CUDA 13.4 packages"):
        state.install()
    assert not (state.artifacts / "nvcc-fingerprint").exists()


@pytest.mark.parametrize(
    ("root", "name"),
    [
        ("artifacts", "fingerprint"),
        ("artifacts", "nvcc.py"),
        ("artifacts", "nvcc_common.py"),
        ("artifacts", "nvcc_warm.py"),
        ("cuda", "bin/ptxas"),
        ("cuda", "bin/nvlink"),
        ("cuda", "bin/fatbinary"),
        ("cuda", "nvvm/libdevice/libdevice.10.bc"),
    ],
)
def test_related_toolchain_and_wrapper_changes_invalidate_cache_generation(
    installation, root, name
):
    state = installation
    state.install()
    before = (state.artifacts / "nvcc-fingerprint").read_text()
    path = getattr(state, root) / name
    path.write_bytes(path.read_bytes() + b" changed")
    state.install()
    assert (state.artifacts / "nvcc-fingerprint").read_text() != before
