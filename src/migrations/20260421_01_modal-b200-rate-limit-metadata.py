"""
Track requested GPUs on submission rows so GPU-specific rate limits can apply before queueing.
"""

from yoyo import step

__depends__ = {"20260318_01_ban-user"}

steps = [
    step(
        """
        ALTER TABLE leaderboard.submission
        ADD COLUMN requested_gpus TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
        """,
        """
        ALTER TABLE leaderboard.submission
        DROP COLUMN requested_gpus;
        """,
    ),
    step(
        """
        CREATE INDEX leaderboard_submission_requested_gpus_idx
        ON leaderboard.submission USING GIN (requested_gpus);
        """,
        """
        DROP INDEX leaderboard.leaderboard_submission_requested_gpus_idx;
        """,
    ),
]
