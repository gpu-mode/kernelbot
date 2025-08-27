"""
website_submission
"""

from yoyo import step

__depends__ = {'20250728_01_Q3jso-fix-code-table'}

# noqa: C901
steps = [
    step(
        "ALTER TABLE leaderboard.user_info "
        "ADD COLUMN IF NOT EXISTS web_auth_id VARCHAR(255) DEFAULT NULL;"
    ),
    step("""
        CREATE TABLE IF NOT EXISTS leaderboard.submission_job_status (
            id              SERIAL PRIMARY KEY,
            submission_id   INTEGER NOT NULL
                            REFERENCES leaderboard.submission(id)
                            ON DELETE CASCADE,
            status          VARCHAR(255) DEFAULT NULL,           -- status of the job
            error           TEXT DEFAULT NULL,                    -- error details if failed
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),   -- creation timestamp
            last_heartbeat  TIMESTAMPTZ DEFAULT NULL,             -- updated periodically by worker
            CONSTRAINT uq_submission_job_status_submission_id
                UNIQUE (submission_id)                            -- one-to-one with submission
        );
    """),
]
