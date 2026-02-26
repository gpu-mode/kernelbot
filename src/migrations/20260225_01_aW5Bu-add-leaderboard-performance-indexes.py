"""
add-leaderboard-performance-indexes
"""

from yoyo import step

__depends__ = {"20260108_01_gzSm3-add-submission-status"}

steps = [
    # FK index on runs.submission_id
    # Used in every ranking query (JOIN runs ON submission_id = s.id)
    step(
        "CREATE INDEX IF NOT EXISTS idx_runs_submission_id ON leaderboard.runs(submission_id)",
        "DROP INDEX IF EXISTS leaderboard.idx_runs_submission_id",
    ),
    # FK index on submission.leaderboard_id
    # Every query filters WHERE leaderboard_id = X
    step(
        "CREATE INDEX IF NOT EXISTS idx_submission_leaderboard_id ON leaderboard.submission(leaderboard_id)",
        "DROP INDEX IF EXISTS leaderboard.idx_submission_leaderboard_id",
    ),
    # FK index on submission.user_id
    # Used in user trend, submission history, and user search queries
    step(
        "CREATE INDEX IF NOT EXISTS idx_submission_user_id ON leaderboard.submission(user_id)",
        "DROP INDEX IF EXISTS leaderboard.idx_submission_user_id",
    ),
    # Partial composite index for ranking queries
    # Covers WHERE NOT r.secret AND r.score IS NOT NULL AND r.passed
    # Includes (submission_id, runner, score) for JOIN, PARTITION BY, ORDER BY
    step(
        "CREATE INDEX IF NOT EXISTS idx_runs_ranking"
        " ON leaderboard.runs(submission_id, runner, score)"
        " WHERE NOT secret AND score IS NOT NULL AND passed",
        "DROP INDEX IF EXISTS leaderboard.idx_runs_ranking",
    ),
]
