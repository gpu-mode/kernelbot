"""
Adds an exclude column to indicate that a milestone is not compatible with certain GPUs
"""

from yoyo import step

__depends__ = {"20250725_01_hwite-add-milestone-table"}

steps = [
    step(
        "ALTER TABLE leaderboard.milestones ADD COLUMN exclude_gpus TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE leaderboard.milestones DROP COLUMN exclude_gpus;",
    )
]
