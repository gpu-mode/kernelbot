"""
add_is_banned_to_user_info
"""

from yoyo import step

__depends__ = {'20260317_01_rate-limits'}

steps = [
    step(
        # forward
        """
        ALTER TABLE leaderboard.user_info
        ADD COLUMN is_banned BOOLEAN NOT NULL DEFAULT FALSE
        """,
        # backward
        """
        ALTER TABLE leaderboard.user_info
        DROP COLUMN is_banned;
        """
    )
]
