"""
leaderboard-rate-limits
"""

from yoyo import step

__depends__ = {"20260108_01_gzSm3-add-submission-status"}

steps = [
    step(
        "ALTER TABLE leaderboard.gpu_type ADD COLUMN rate_limit_seconds INTEGER DEFAULT NULL",
        "ALTER TABLE leaderboard.gpu_type DROP COLUMN rate_limit_seconds",
    ),
]
