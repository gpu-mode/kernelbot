"""
submission-add-user-name
"""

from yoyo import step

__depends__ = {"20250221_01_GA8ro-submission-collection"}

steps = [step("ALTER TABLE leaderboard.submission ADD COLUMN user_name TEXT")]
