"""
Add per-leaderboard rate limiting configuration and track mode category on submissions.
"""

from yoyo import step

__depends__ = {"20260313_01_invite-multi-leaderboard"}

steps = [
    step(
        """
        CREATE TABLE leaderboard.rate_limit (
            id SERIAL PRIMARY KEY,
            leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
            mode_category TEXT NOT NULL CHECK (mode_category IN ('test', 'leaderboard')),
            max_submissions_per_hour INTEGER NOT NULL CHECK (max_submissions_per_hour > 0),
            UNIQUE (leaderboard_id, mode_category)
        );
        """,
        """
        DROP TABLE leaderboard.rate_limit;
        """,
    ),
    step(
        """
        ALTER TABLE leaderboard.submission ADD COLUMN mode_category TEXT;
        """,
        """
        ALTER TABLE leaderboard.submission DROP COLUMN mode_category;
        """,
    ),
]
