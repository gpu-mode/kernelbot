"""
add-task-version
"""

from yoyo import step

__depends__ = {'20260318_01_ban-user'}


steps = [
    step(
        """
        ALTER TABLE leaderboard.leaderboard
        ADD COLUMN task_version INT NOT NULL DEFAULT 1;
        """,
        """
        ALTER TABLE leaderboard.leaderboard
        DROP COLUMN task_version;
        """
    ),
    step(
        """
        ALTER TABLE leaderboard.runs
        ADD COLUMN task_version INT NOT NULL DEFAULT 1;
        """,
        """
        ALTER TABLE leaderboard.runs
        DROP COLUMN task_version;
        """
    ),
    # Update the partial index to include task_version for efficient filtering
    step(
        """
        DROP INDEX IF EXISTS leaderboard.idx_runs_valid_scores;
        CREATE INDEX idx_runs_valid_scores
        ON leaderboard.runs (submission_id, runner, score, task_version)
        WHERE NOT secret AND score IS NOT NULL AND passed;
        """,
        """
        DROP INDEX IF EXISTS leaderboard.idx_runs_valid_scores;
        CREATE INDEX idx_runs_valid_scores
        ON leaderboard.runs (submission_id, runner, score)
        WHERE NOT secret AND score IS NOT NULL AND passed;
        """
    ),
]
