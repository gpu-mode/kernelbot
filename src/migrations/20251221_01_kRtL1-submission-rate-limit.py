"""
submission_rate_limit
"""

from yoyo import step

__depends__ = {"20251106_01_kOjGy-draft-code-editor"}

steps = [
    step(
        """
        CREATE TABLE IF NOT EXISTS leaderboard.submission_rate_limit_settings (
            id                      BOOLEAN PRIMARY KEY DEFAULT TRUE,
            default_rate_per_minute DOUBLE PRECISION DEFAULT NULL,
            capacity                DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    ),
    step(
        """
        INSERT INTO leaderboard.submission_rate_limit_settings (id, default_rate_per_minute, capacity)
        VALUES (TRUE, NULL, 1.0)
        ON CONFLICT (id) DO NOTHING;
        """  # noqa: E501
    ),
    step(
        """
        CREATE TABLE IF NOT EXISTS leaderboard.submission_rate_limit_user (
            user_id         TEXT PRIMARY KEY,
            rate_per_minute DOUBLE PRECISION DEFAULT NULL,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    ),
    step(
        """
        CREATE TABLE IF NOT EXISTS leaderboard.submission_rate_limit_state (
            user_id     TEXT PRIMARY KEY,
            tokens      DOUBLE PRECISION NOT NULL,
            last_refill TIMESTAMPTZ NOT NULL
        );
        """
    ),
]
