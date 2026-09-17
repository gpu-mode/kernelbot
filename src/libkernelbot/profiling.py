"""Validated, request-scoped profiling options shared by API and GPU runners."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ProfileOptions:
    benchmark_index: int | None = None
    ncu_kernel_name: str | None = None
    ncu_kernel_name_base: Literal["function", "demangled", "mangled"] | None = None
    ncu_launch_count: int | None = None

    def __post_init__(self):
        for name, minimum in [("benchmark_index", 0), ("ncu_launch_count", 1)]:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.ncu_kernel_name_base not in (None, "function", "demangled", "mangled"):
            raise ValueError("ncu_kernel_name_base must be function, demangled, or mangled")
        if self.ncu_kernel_name is not None and (
            not isinstance(self.ncu_kernel_name, str)
            or not self.ncu_kernel_name
            or len(self.ncu_kernel_name) > 1024
            or "\x00" in self.ncu_kernel_name
        ):
            raise ValueError("ncu_kernel_name must contain 1..1024 characters without NUL")

    def validate_task(self, benchmarks: list, multi_gpu: bool):
        if multi_gpu:
            raise ValueError("NCU profiling requires a single GPU")
        if not benchmarks:
            raise ValueError("This task has no benchmarks to profile")
        if self.benchmark_index is not None and self.benchmark_index >= len(benchmarks):
            raise ValueError(f"benchmark_index must be between 0 and {len(benchmarks) - 1}")

    def to_dict(self) -> dict:
        return {name: value for name, value in asdict(self).items() if value is not None}
