"""
queryindex
"""

from yoyo import step

__depends__ = {'20260225_01_aW5Bu-add-leaderboard-performance-indexes'}


steps = [
    # Most critical: partial composite index on runs
    step(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_valid_scores
        ON leaderboard.runs (submission_id, runner, score)
        WHERE NOT secret AND score IS NOT NULL AND passed;
        """,
        """
        DROP INDEX CONCURRENTLY IF EXISTS leaderboard.idx_runs_valid_scores;
        """
    ),
    step(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_submission_leaderboard_id
        ON leaderboard.submission (leaderboard_id);
        """,
        """
        DROP INDEX CONCURRENTLY IF EXISTS leaderboard.idx_submission_leaderboard_id;
        """
    ),
    step(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_submission_user_id
        ON leaderboard.submission (user_id);
        """,
        """
        DROP INDEX CONCURRENTLY IF EXISTS leaderboard.idx_submission_user_id;
        """
    ),
]
