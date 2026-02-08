"""
add-user-rate-limits
"""

from yoyo import step

__depends__ = {'20260108_01_gzSm3-add-submission-status'}

steps = [
    step(
        # forward
        """
        CREATE TABLE leaderboard.user_rate_limits (
            user_id TEXT PRIMARY KEY REFERENCES leaderboard.user_info(id),
            max_submissions_per_hour INTEGER,
            max_submissions_per_day INTEGER,
            note TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """,
        # backward
        """
        DROP TABLE leaderboard.user_rate_limits;
        """
    )
]
