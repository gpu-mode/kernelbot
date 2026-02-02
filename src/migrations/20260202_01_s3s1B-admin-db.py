"""
admin_db
"""

from yoyo import step

__depends__ = {'20260108_01_gzSm3-add-submission-status'}

steps = [
    step(
        """
        ALTER TABLE leaderboard.leaderboard
          ADD COLUMN status TEXT DEFAULT 'active' NOT NULL,
          ADD COLUMN config JSONB,
          ADD COLUMN start_at TIMESTAMPTZ,
          ADD COLUMN last_modified TIMESTAMPTZ;
        """,
        """
        ALTER TABLE leaderboard.leaderboard
          DROP COLUMN IF EXISTS last_modified,
          DROP COLUMN IF EXISTS start_at,
          DROP COLUMN IF EXISTS config,
          DROP COLUMN IF EXISTS status;
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.runs
          ADD COLUMN status TEXT DEFAULT 'active' NOT NULL;
        """,
        """
        ALTER TABLE leaderboard.runs
          DROP COLUMN IF EXISTS status;
        """
    ),
]
