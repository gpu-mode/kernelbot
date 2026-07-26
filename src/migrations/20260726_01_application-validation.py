"""
Persist problem-owned application validation results for leaderboard submissions.
"""

from yoyo import step

__depends__ = {"20260319_01_allow-zero-rate-limits"}

steps = [
    step(
        """
        CREATE TABLE leaderboard.validation_sweep (
            id BIGSERIAL PRIMARY KEY,
            leaderboard_id INTEGER NOT NULL
                REFERENCES leaderboard.leaderboard(id) ON DELETE CASCADE,
            gpu_type TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            scheduled_for DATE,
            status TEXT NOT NULL
                CHECK (status IN ('running', 'completed', 'failed')),
            top_k INTEGER NOT NULL CHECK (top_k > 0),
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            error TEXT,
            UNIQUE (leaderboard_id, gpu_type, contract_version, scheduled_for)
        );

        CREATE TABLE leaderboard.submission_validation (
            id BIGSERIAL PRIMARY KEY,
            submission_id INTEGER NOT NULL
                REFERENCES leaderboard.submission(id) ON DELETE CASCADE,
            gpu_type TEXT NOT NULL,
            contract_name TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('completed', 'failed')),
            passed_shapes INTEGER NOT NULL DEFAULT 0 CHECK (passed_shapes >= 0),
            total_shapes INTEGER NOT NULL DEFAULT 0 CHECK (total_shapes >= 0),
            fully_validated BOOLEAN NOT NULL DEFAULT FALSE,
            geomean_sync_wall_speedup DOUBLE PRECISION,
            result JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT,
            checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (submission_id, gpu_type, contract_version)
        );

        CREATE INDEX submission_validation_latest_idx
            ON leaderboard.submission_validation
            (submission_id, gpu_type, checked_at DESC);
        """,
        """
        DROP TABLE leaderboard.submission_validation;
        DROP TABLE leaderboard.validation_sweep;
        """,
    )
]
