from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    name: str  # e.g., "A100", "H100", "4090"
    count: int  # Number of GPUs
    memory: int  # Memory in GB
    provider: Optional[str] = None  # e.g., "AWS", "Local", "Lambda Labs"

@dataclass
class KernelInfo:
    name: str  # e.g., "GEMM FP16", "MatMul", "Flash Attention"
    language: str  # e.g., "CUDA", "HIP", "Triton"
    problem_size: str  # e.g., "1024x1024", "batch_size=32,seq_len=512"
    description: str  # Detailed description of the kernel and its parameters
    runtime: float  # Runtime in milliseconds
    framework: Optional[str] = None  # e.g., "PyTorch", "TensorFlow", "Custom"
    version: Optional[str] = None  # Framework/compiler version

@dataclass
class LeaderboardEntry:
    username: str
    timestamp: datetime
    hardware: HardwareInfo
    kernel: KernelInfo
    metrics: Dict[str, float]
    run_id: str

class Leaderboard:
    def __init__(self):
        """Initialize an empty leaderboard"""
        self.entries: Dict[str, List[LeaderboardEntry]] = {}  # Hardware -> Entries mapping
    
    def add_entry(self, username: str, hardware: HardwareInfo, kernel: KernelInfo, 
                 metrics: Dict[str, float], run_id: str) -> None:
        """Add a new entry to the appropriate hardware leaderboard"""
        hardware_key = f"{hardware.name}"
        if hardware_key not in self.entries:
            self.entries[hardware_key] = []
            
        entry = LeaderboardEntry(
            username=username,
            timestamp=datetime.now(timezone.utc),
            hardware=hardware,
            kernel=kernel,
            metrics=metrics,
            run_id=run_id
        )
        self.entries[hardware_key].append(entry)
        logger.info(f"Added new leaderboard entry for {username} on {hardware_key}")

    async def format_discord_message(self, hardware_name: str) -> str:
        if hardware_name not in self.entries:
            return "```md\n# No entries for {hardware_name} yet!\n```"
        
        entries = self.entries[hardware_name]
        
        # Group entries by username
        user_entries = {}
        for entry in entries:
            if entry.username not in user_entries:
                user_entries[entry.username] = []
            user_entries[entry.username].append(entry)
        
        output = [
            "```md",
            f"# {hardware_name} CUDA Kernel Benchmarks",
            "===============================",
        ]
        
        # Sort users by their best performing kernel (lowest runtime)
        sorted_users = sorted(
            user_entries.items(),
            key=lambda x: min(e.kernel.runtime for e in x[1])
        )
        
        for rank, (username, user_kernels) in enumerate(sorted_users, 1):
            rank_display = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            output.extend([
                f"\n## {rank_display} {username}'s Benchmarks",
                "| Kernel | Runtime(ms) | TFLOPS | BW(GB/s) |",
                "|--------|-------------|---------|-----------|"
            ])
            
            # Sort user's kernels by runtime
            sorted_kernels = sorted(user_kernels, key=lambda x: x.kernel.runtime)
            for entry in sorted_kernels:
                metrics = entry.metrics
                output.append(
                    f"| {entry.kernel.name:<20} | {entry.kernel.runtime:>8.2f} | "
                    f"{metrics.get('Throughput in TFLOPS', 0):>7.1f} | "
                    f"{metrics.get('Memory Bandwidth in GB/s', 0):>9.1f} |"
                )
                output.append(f"Problem Size: {entry.kernel.problem_size}")
                output.append(f"Description: {entry.kernel.description}")
                output.append("|---------")
        
        output.append("```")
        return "\n".join(output)

    def get_entry_by_run_id(self, run_id: str) -> Optional[LeaderboardEntry]:
        """Find an entry by its GitHub run ID"""
        for entries in self.entries.values():
            for entry in entries:
                if entry.run_id == run_id:
                    return entry
        return None

    def get_available_hardware(self) -> List[str]:
        """Returns list of hardware types that have entries"""
        return list(self.entries.keys())
