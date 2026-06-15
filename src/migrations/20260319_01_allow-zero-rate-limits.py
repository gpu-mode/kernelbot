"""
Allow zero-submission rate limits.
"""

from yoyo import step

__depends__ = {"20260318_01_ban-user"}

steps = [
    step(
        """
        ALTER TABLE leaderboard.rate_limit
        DROP CONSTRAINT rate_limit_max_submissions_per_hour_check;

        ALTER TABLE leaderboard.rate_limit
        ADD CONSTRAINT rate_limit_max_submissions_per_hour_check
        CHECK (max_submissions_per_hour >= 0);
        """,
        """
        ALTER TABLE leaderboard.rate_limit
        DROP CONSTRAINT rate_limit_max_submissions_per_hour_check;

        ALTER TABLE leaderboard.rate_limit
        ADD CONSTRAINT rate_limit_max_submissions_per_hour_check
        CHECK (max_submissions_per_hour > 0);
        """,
    )
]
