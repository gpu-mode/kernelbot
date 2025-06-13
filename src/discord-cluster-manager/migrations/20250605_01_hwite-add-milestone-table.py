"""
Add milestone table for better milestone tracking
"""

from yoyo import step

__depends__ = {"20250506_01_38PkG-add-index-on-runs-runner-score"}  # Update to latest migration

steps = [
    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.milestones (
             id SERIAL PRIMARY KEY,
             leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id),
             milestone_name TEXT NOT NULL,
             filename TEXT NOT NULL,
             description TEXT,
             created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
             UNIQUE(leaderboard_id, milestone_name)
         )
         """),
    step("CREATE INDEX ON leaderboard.milestones (leaderboard_id)"),
    step("""
         CREATE TABLE IF NOT EXISTS leaderboard.milestone_runs (
             id SERIAL PRIMARY KEY,
             milestone_id INTEGER NOT NULL REFERENCES leaderboard.milestones(id),
             submission_id INTEGER NOT NULL REFERENCES leaderboard.submission(id),
             run_id INTEGER NOT NULL REFERENCES leaderboard.runs(id),
             created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
         )
         """),
    step("CREATE INDEX ON leaderboard.milestone_runs (milestone_id)"),
    step("CREATE INDEX ON leaderboard.milestone_runs (submission_id)"),
]
