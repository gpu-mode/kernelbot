"""
add-submission-audit
"""

from yoyo import step

__depends__ = {"20260225_01_aW5Bu-add-leaderboard-performance-indexes"}

steps = [
    step(
        """
        CREATE TABLE leaderboard.submission_audit (
            id SERIAL PRIMARY KEY,
            submission_id INTEGER NOT NULL REFERENCES leaderboard.submission(id) ON DELETE CASCADE,
            is_cheating BOOLEAN NOT NULL,
            explanation TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reviewed BOOLEAN NOT NULL DEFAULT FALSE,
            UNIQUE(submission_id)
        )
        """,
        """
        DROP TABLE IF EXISTS leaderboard.submission_audit
        """,
    ),
]
