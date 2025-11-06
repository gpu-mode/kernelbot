"""
draft_code_editor
"""

from yoyo import step

__depends__ = {'20250822_01_UtXzl-website-submission'}

steps = [
    step("""
        CREATE TABLE IF NOT EXISTS leaderboard.draft_code (
            id SERIAL PRIMARY KEY,
            leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
            type TEXT NOT NULL DEFAULT 'general',
            code BYTEA NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
]
