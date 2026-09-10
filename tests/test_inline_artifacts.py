"""CPU-only contract tests; the Modal prototype exercises real compilation/loading."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from libkernelbot.inline_artifacts import install, pack_artifacts, unpack_artifacts


class InlineArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "compiled.so"
        self.binary.write_bytes(b"fake compiled extension")
        self.builds = 0

        def load_inline(
            name,
            cpp_sources,
            cuda_sources=None,
            extra_cuda_cflags=None,
            extra_ldflags=None,
            is_python_module=True,
            verbose=False,
            build_directory=None,
        ):
            self.builds += 1
            return types.SimpleNamespace(__name__=name, __file__=str(self.binary))

        self.original = load_inline
        self.extension = types.SimpleNamespace(load_inline=load_inline)
        torch = types.SimpleNamespace(
            __version__="test",
            version=types.SimpleNamespace(cuda="13.3"),
            _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        )
        modules = patch.dict(
            sys.modules,
            {
                "torch": torch,
                "torch.utils": types.SimpleNamespace(cpp_extension=self.extension),
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(
            os.environ,
            {
                "KERNELBOT_INLINE_MODE": "capture",
                "KERNELBOT_INLINE_ARTIFACTS": str(self.root / "artifacts"),
                "KERNELBOT_INLINE_WORKDIR": str(self.root),
                "TORCH_CUDA_ARCH_LIST": "7.5",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        dependencies = patch("libkernelbot.inline_artifacts._dependencies", return_value=[])
        dependencies.start()
        self.addCleanup(dependencies.stop)
        install()
        self.extension.load_inline("example", "source", cuda_sources="cuda")

    def replay(self):
        self.extension.load_inline = self.original
        os.environ["KERNELBOT_INLINE_MODE"] = "replay"
        install()

    def test_changed_source_and_flags_do_not_reuse_binary(self):
        self.replay()
        for args in ({"cpp_sources": "changed"}, {"extra_cuda_cflags": ["-O3"]}):
            request = {"name": "example", "cpp_sources": "source", "cuda_sources": "cuda", **args}
            with self.assertRaisesRegex(RuntimeError, "No CPU-built artifact"):
                self.extension.load_inline(**request)
        self.assertEqual(self.builds, 1)

    def test_changed_architecture_does_not_reuse_binary(self):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        self.replay()
        with self.assertRaisesRegex(RuntimeError, "No CPU-built artifact"):
            self.extension.load_inline("example", "source", cuda_sources="cuda")

    def test_replay_loads_saved_module_once_without_compiling(self):
        self.replay()
        module = types.SimpleNamespace()
        spec = Mock()
        with (
            patch("importlib.util.spec_from_file_location", return_value=spec) as make_spec,
            patch("importlib.util.module_from_spec", return_value=module),
        ):
            for _ in range(2):
                self.assertIs(
                    self.extension.load_inline("example", "source", cuda_sources="cuda"), module
                )
        self.assertEqual(make_spec.call_args.args[0], "example")
        self.assertEqual(make_spec.call_args.args[1].read_bytes(), self.binary.read_bytes())
        spec.loader.exec_module.assert_called_once_with(module)
        self.assertEqual(self.builds, 1)

    def test_wrong_manifest_is_rejected_before_loading(self):
        manifest_path = next((self.root / "artifacts").glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime"]["cuda"] = "different"
        manifest_path.write_text(json.dumps(manifest))
        self.replay()
        with self.assertRaisesRegex(RuntimeError, "manifest does not match"):
            self.extension.load_inline("example", "source", cuda_sources="cuda")

    def test_corrupted_artifact_is_rejected_before_loading(self):
        next((self.root / "artifacts").glob("*/extension.so")).write_bytes(b"corrupted")
        self.replay()
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.extension.load_inline("example", "source", cuda_sources="cuda")

    def test_ninja_is_disabled_during_replay(self):
        self.replay()
        with self.assertRaisesRegex(RuntimeError, "GPU compilation is disabled"):
            self.extension._run_ninja_build()

    def test_native_mode_falls_back_to_original_compiler(self):
        self.extension.load_inline = self.original
        os.environ["KERNELBOT_INLINE_MODE"] = "auto"
        install()
        self.extension.load_inline("example", "changed source", cuda_sources="cuda")
        self.assertEqual(self.builds, 2)

    def test_environment_changes_after_import_do_not_reuse_binary(self):
        self.replay()
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        with self.assertRaisesRegex(RuntimeError, "No CPU-built artifact"):
            self.extension.load_inline("example", "source", cuda_sources="cuda")

    def test_custom_linker_inputs_are_left_to_the_gpu(self):
        with self.assertRaisesRegex(ValueError, "custom linker inputs"):
            self.extension.load_inline("linked", "source", extra_ldflags=["-lcustom"])
        self.assertEqual(self.builds, 1)
        self.extension.load_inline = self.original
        os.environ["KERNELBOT_INLINE_MODE"] = "auto"
        install()
        self.extension.load_inline("linked", "source", extra_ldflags=["-lcustom"])
        self.assertEqual(self.builds, 2)

    def test_transfer_contains_only_binary_and_manifest(self):
        artifacts = pack_artifacts(self.root / "artifacts")
        self.assertEqual(len(artifacts), 2)
        destination = self.root / "received"
        unpack_artifacts(destination, artifacts)
        self.assertEqual(pack_artifacts(destination), artifacts)

    def test_transfer_rejects_path_escape(self):
        with self.assertRaisesRegex(ValueError, "Invalid artifact path"):
            unpack_artifacts(self.root / "received", {"../escape.so": b"binary"})
