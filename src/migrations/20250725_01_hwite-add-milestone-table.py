"""
Add milestone table for better milestone tracking
"""

from yoyo import step

__depends__ = {"20250617_01_c5mrF-task-split"}  # Update to latest migration

steps = [
    step(
        """
         CREATE TABLE IF NOT EXISTS leaderboard.milestones (
             id SERIAL PRIMARY KEY,
             leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
             name TEXT NOT NULL,
             code TEXT NOT NULL,
             description TEXT,
             created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
             UNIQUE(leaderboard_id, name)
         )
         """,
        "DROP TABLE leaderboard.milestones",
    ),
    step("CREATE INDEX ON leaderboard.milestones (leaderboard_id)"),
    # add alternative ID column that references milestones;
    # we don't really care about being careful preserving milestone
    # runs, so we can simply DELETE CASCADE
    step(
        """
        ALTER TABLE leaderboard.runs
        ADD COLUMN milestone_id INTEGER REFERENCES leaderboard.milestones(id) ON DELETE CASCADE;
        """,
        "ALTER TABLE leaderboard.runs DROP COLUMN milestone_id;",
    ),
    # as we now have two possible ids, exactly one of them can be NULL
    step(
        "ALTER TABLE leaderboard.runs ALTER COLUMN submission_id DROP NOT NULL;",
        """
         DELETE FROM leaderboard.runs WHERE submission_id IS NULL;
         ALTER TABLE leaderboard.runs ALTER COLUMN submission_id SET NOT NULL;
         """,
    ),
    step(
        """
        ALTER TABLE leaderboard.runs
        ADD CONSTRAINT runs_single_parent CHECK (
            (submission_id IS NOT NULL AND milestone_id IS NULL) OR
            (submission_id IS NULL AND milestone_id IS NOT NULL)
        );
        """,
        "ALTER TABLE leaderboard.runs DROP CONSTRAINT runs_single_parent;",
    ),
    # ensure we have fast indexing for regular submissions
    step(
        """
        CREATE INDEX IF NOT EXISTS runs_submission_id_idx ON leaderboard.runs(submission_id)
        WHERE submission_id IS NOT NULL;
        """,
        "DROP INDEX IF EXISTS leaderboard.runs_submission_id_idx",
    ),
]
