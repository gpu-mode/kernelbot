"""
remember forum_id for leaderboard:
    makes it much easier for us to keep track of the corresponding thread; we don't have to rely on name matching
mark leaderboards as active/inactive:
    submissions reference leaderboards, so we cannot delete a leaderboard without also deleting all its submissions.
    instead, we can just mark a leaderboard as not active, and hide it everywhere, but keep it and its submissions
    around
"""

from yoyo import step

__depends__ = {'20250304_01_DzORz-collect-system-information-for-each-run'}

steps = [
    step(
        "ALTER TABLE leaderboard.leaderboard ADD COLUMN forum_id BIGINT NOT NULL DEFAULT -1",
    ),
    step("ALTER TABLE leaderboard.leaderboard ALTER COLUMN forum_id DROP DEFAULT;"),
    step("ALTER TABLE leaderboard.leaderboard ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE"),
]
