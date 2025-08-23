"""
website_submission
"""

from yoyo import step

__depends__ = {'20250728_01_Q3jso-fix-code-table'}

steps = [
    step(
        "ALTER TABLE leaderboard.user_info ADD COLUMN IF NOT EXISTS web_auth_id VARCHAR(255) DEFAULT NULL;"
    )
]
