"""Pinned private NVIDIA compiler drop used by the Modal runner image."""

RELEASE = "nvidia-pch-13.4.99-20261017"
BINARY_SHA256 = {
    "nvcc": "254c308d113e2cdd0159ddfc02b54630b36be2013c0b928b6513e4bae723a298",
    "cudafe++": "f64ef97248879dccf4cd92aef02d56a8ea915fbc20abd288e63dd9c199f24442",
    "nvvm/bin/cicc": "9b890cf57ba86bc2d3c56714cf24e7c5fe52d0919c3b13512a3eea75ac2c3050",
    "nvvm-next/bin/cicc": "5cebc486361825fe2625031922b8b8d481556cc4051006f1abd0c8f440e8a1c6",
}


def install_nvcc_toolchain(release: str, expected_hashes: dict[str, str]):  # noqa: C901
    """Image-build step: consume a private read-only volume, never submission input."""
    import hashlib
    import json
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    source = Path("/nvcc-toolchain-input") / release
    cuda = Path("/usr/local/cuda")
    targets = {
        "nvcc": "bin/nvcc",
        "cudafe++": "bin/cudafe++",
        "nvvm/bin/cicc": "nvvm/bin/cicc",
        "nvvm-next/bin/cicc": "nvvm-next/bin/cicc",
    }
    if expected_hashes.keys() != targets.keys():
        raise ValueError("Expected checksums for exactly the four NVIDIA compiler binaries")
    for name, expected in expected_hashes.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"NVIDIA compiler checksum mismatch: {name}")
    for name, target_name in targets.items():
        target = cuda / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        target.chmod(0o755)

    # All supported Modal runner GPUs are SM75 or newer. The vendor profile
    # otherwise keeps selecting the old backend despite the second binary.
    profile_path = cuda / "bin/nvcc.profile"
    profile, count = re.subn(
        r"^CICC_PATH\s*=.*$",
        "CICC_PATH        = $(TOP)/nvvm-next/bin",
        profile_path.read_text(),
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Expected exactly one CICC_PATH in the CUDA compiler profile")
    profile_path.write_text(profile)

    compiler = str(cuda / "bin/nvcc")
    version = subprocess.check_output([compiler, "--version"], text=True)
    if "V13.4.99" not in version:
        raise ValueError(f"Unexpected NVIDIA compiler version: {version}")
    smoke = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_file = root / "smoke.cu"
        source_file.write_text("__global__ void kernel(int* x) { x[0] = 7; }\n")
        for arch in ("75", "80", "89", "90a", "100", "100a"):
            command = [compiler, f"-arch=sm_{arch}", "-c", str(source_file), "-o"]
            command.append(str(root / f"sm_{arch}.o"))
            dryrun = subprocess.check_output(
                [*command, "--dryrun"], stderr=subprocess.STDOUT, text=True
            )
            if "nvvm-next/bin" not in dryrun:
                raise ValueError(f"Wrong CUDA compiler backend for SM{arch}")
            subprocess.run(command, check=True, timeout=60)
            smoke.append(arch)

    package_output = subprocess.check_output(
        [
            "dpkg-query",
            "--show",
            "--showformat=${Package}\t${Version}\t${Status}\n",
            "*13-4*",
        ],
        text=True,
    )
    cuda_packages = {}
    for line in package_output.splitlines():
        name, package_version, status = line.split("\t", 2)
        if status == "install ok installed":
            cuda_packages[name] = package_version
    if not cuda_packages:
        raise ValueError("No installed CUDA 13.4 packages found for the toolchain fingerprint")

    artifact_root = Path("/opt/kernelbot-pch")
    files = [
        *(cuda / name for name in targets.values()),
        profile_path,
        *(cuda / "bin" / name for name in ("ptxas", "nvlink", "fatbinary")),
        cuda / "nvvm/libdevice/libdevice.10.bc",
        *(artifact_root / name for name in ("nvcc.py", "nvcc_common.py", "nvcc_warm.py")),
    ]
    metadata = {
        "version": 1,
        "release": release,
        "nvcc_version": version,
        "smoke_architectures": smoke,
        "cuda_packages": cuda_packages,
        "headers_fingerprint": (artifact_root / "fingerprint").read_text().strip(),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (artifact_root / "nvcc-toolchain.json").write_text(
        json.dumps({**metadata, "fingerprint": fingerprint}, indent=2) + "\n"
    )
    (artifact_root / "nvcc-fingerprint").write_text(fingerprint + "\n")
    print(f"Installed {release}; toolchain fingerprint {fingerprint}", flush=True)
