"""
add_submission_status
"""

from yoyo import step

__depends__ = {'20251106_01_kOjGy-draft-code-editor'}

steps = [
    step(
        # forward
        """
        ALTER TABLE leaderboard.submission
        ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
        """,
        # backward for rollback if yoyo apply fails for any reason
        """
        ALTER TABLE leaderboard.submission
        DROP COLUMN status;
        """
    )
]
