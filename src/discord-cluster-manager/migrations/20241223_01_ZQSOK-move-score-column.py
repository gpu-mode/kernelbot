"""
This migration adds a score column to runinfo, and drops the score column in
submission.
"""

from yoyo import step

__depends__ = {'20241222_01_ELxU5-add-gpu-types'}

steps = [
    step(
        "ALTER TABLE leaderboard.runinfo ADD COLUMN score NUMERIC NOT NULL",
        "ALTER TABLE leaderboard.runinfo DROP COLUMN score"
    ),

    step(
        "ALTER TABLE leaderboard.submission DROP COLUMN score",
        """
        ALTER TABLE leaderboard.submission
        ADD COLUMN score NUMERIC NOT NULL DEFAULT 0
        """,
    )
]
