"""
add-leaderboard-performance-indexes
"""

from yoyo import step

__depends__ = {"20260108_01_gzSm3-add-submission-status"}

steps = [
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
]
