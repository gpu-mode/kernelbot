"""
submission-add-user-name
"""

from yoyo import step

__depends__ = {"20250221_01_GA8ro-submission-collection"}

steps = [
    step("""
        CREATE TABLE leaderboard.user_info (
            user_id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL
        )
    """),
    step("""
        INSERT INTO leaderboard.user_info (user_id, user_name)
        SELECT DISTINCT user_id, 'Unknown User' FROM leaderboard.submission
    """),
    step("""
        ALTER TABLE leaderboard.submission
        ADD CONSTRAINT fk_user_info
        FOREIGN KEY (user_id) REFERENCES leaderboard.user_info(user_id)
    """),
]
