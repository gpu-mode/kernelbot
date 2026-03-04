"""
add-allowed-users
"""

from yoyo import step

__depends__ = {"20260226_01_WgYAV-queryindex"}


steps = [
    step(
        """
        ALTER TABLE leaderboard.leaderboard
        ADD COLUMN allowed_users TEXT[] DEFAULT NULL;
        """,
        """
        ALTER TABLE leaderboard.leaderboard
        DROP COLUMN IF EXISTS allowed_users;
        """,
    ),
]
