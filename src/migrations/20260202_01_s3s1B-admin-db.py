"""
admin_db
"""

from yoyo import step

__depends__ = {'20260108_01_gzSm3-add-submission-status'}

steps = [
    step(
        """
        ALTER TABLE leaderboard.leaderboard
          ADD COLUMN status TEXT DEFAULT 'active',
          ADD COLUMN config JSONB,
          ADD COLUMN start_time TIMESTAMPTZ,
          ADD COLUMN last_modified TIMESTAMPTZ;
        """,
        """
        ALTER TABLE leaderboard.leaderboard
          DROP COLUMN IF EXISTS last_modified,
          DROP COLUMN IF EXISTS start_time,
          DROP COLUMN IF EXISTS config,
          DROP COLUMN IF EXISTS status;
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.runs
          ADD COLUMN status TEXT DEFAULT 'pending';
        """,
        """
        ALTER TABLE leaderboard.runs
          DROP COLUMN IF EXISTS status;
        """
    ),
]
