"""
add-leaderboard-visibility
"""

from yoyo import step

__depends__ = {'20260226_01_WgYAV-queryindex'}


steps = [
    step(
        """
        ALTER TABLE leaderboard.leaderboard
        ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public';
        """,
        """
        ALTER TABLE leaderboard.leaderboard
        DROP COLUMN visibility;
        """
    ),
    step(
        """
        CREATE TABLE leaderboard.leaderboard_invite (
            id SERIAL PRIMARY KEY,
            leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
            code TEXT NOT NULL UNIQUE,
            user_id TEXT,
            claimed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        DROP TABLE leaderboard.leaderboard_invite;
        """
    ),
    step(
        """
        CREATE INDEX idx_leaderboard_invite_leaderboard
        ON leaderboard.leaderboard_invite (leaderboard_id);
        """,
        """
        DROP INDEX IF EXISTS leaderboard.idx_leaderboard_invite_leaderboard;
        """
    ),
    step(
        """
        CREATE INDEX idx_leaderboard_invite_user
        ON leaderboard.leaderboard_invite (user_id)
        WHERE user_id IS NOT NULL;
        """,
        """
        DROP INDEX IF EXISTS leaderboard.idx_leaderboard_invite_user;
        """
    ),
]
