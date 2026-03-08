"""Export competition submissions to Hugging Face datasets as parquet files."""

import io
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from libkernelbot.leaderboard_db import LeaderboardDB
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)

# Explicit schema matching GPUMODE/kernelbot-data nvidia_nvfp4_submissions.parquet
SUBMISSIONS_SCHEMA = pa.schema([
    ("submission_id", pa.int64()),
    ("leaderboard_id", pa.int64()),
    ("problem_name", pa.large_string()),
    ("user_id", pa.large_string()),
    ("user_name", pa.large_string()),
    ("code_id", pa.int64()),
    ("file_name", pa.large_string()),
    ("submission_time", pa.timestamp("us", tz="UTC")),
    ("status", pa.large_string()),
    ("score", pa.float64()),
    ("passed", pa.bool_()),
    ("mode", pa.large_string()),
    ("runner", pa.large_string()),
    ("code", pa.large_string()),
])


def _normalize_deadline(deadline: datetime) -> datetime:
    """Ensure deadlines are timezone-aware before comparing them."""
    if deadline.tzinfo is None:
        return deadline.replace(tzinfo=timezone.utc)
    return deadline


MAX_COMPETITION_HORIZON_DAYS = 365


def get_active_competition_leaderboards(
    leaderboards: list[dict],
    *,
    now: datetime | None = None,
) -> list[dict]:
    """Return leaderboards that belong to real, active competitions.

    Filters out:
    - Expired leaderboards (deadline <= now)
    - Dev leaderboards (name ending with "-dev")
    - Permanent/practice leaderboards (deadline > 1 year from now, e.g. year 2100)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    from datetime import timedelta

    horizon = now + timedelta(days=MAX_COMPETITION_HORIZON_DAYS)

    active_competitions = []
    for leaderboard in leaderboards:
        deadline = _normalize_deadline(leaderboard["deadline"])
        if deadline > now and deadline < horizon and not leaderboard["name"].endswith("-dev"):
            active_competitions.append(leaderboard)
    return active_competitions


def ensure_public_export_allowed(
    db: LeaderboardDB,
    leaderboard_ids: list[int],
    *,
    now: datetime | None = None,
) -> None:
    """Block public exports while any selected leaderboard is still active."""
    if now is None:
        now = datetime.now(timezone.utc)

    selected_ids = set(leaderboard_ids)
    active_names = []
    for leaderboard in db.get_leaderboards():
        if leaderboard["id"] not in selected_ids:
            continue
        deadline = _normalize_deadline(leaderboard["deadline"])
        if deadline > now:
            active_names.append(leaderboard["name"])

    if active_names:
        active_names.sort()
        raise ValueError(
            "Cannot export active leaderboards to the public dataset: "
            + ", ".join(active_names)
        )


def get_hf_export_rows(db: LeaderboardDB, leaderboard_ids: list[int]) -> list[dict]:
    """Fetch deduplicated submissions for export.

    Deduplicates by (leaderboard_id, user_id, code_id, runner), keeping the
    fastest score. Excludes secret runs.
    """
    if not leaderboard_ids:
        return []

    db.cursor.execute(
        """
        WITH ranked AS (
            SELECT
                s.id as submission_id,
                s.leaderboard_id,
                l.name as problem_name,
                s.user_id,
                u.user_name,
                s.code_id,
                s.file_name,
                s.submission_time,
                COALESCE(
                    sjs.status,
                    CASE
                        WHEN s.done AND r.score IS NOT NULL AND r.passed THEN 'succeeded'
                        WHEN s.done THEN 'failed'
                        ELSE s.status
                    END
                ) as status,
                r.score,
                r.passed,
                r.mode,
                r.runner,
                COALESCE(c.old_code, convert_from(c.code, 'UTF8')) as code,
                ROW_NUMBER() OVER (
                    PARTITION BY s.leaderboard_id, s.user_id, s.code_id, r.runner
                    ORDER BY r.score ASC NULLS LAST, s.submission_time ASC
                ) as rn
            FROM leaderboard.submission s
            JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
            LEFT JOIN leaderboard.user_info u ON s.user_id = u.id
            LEFT JOIN leaderboard.submission_job_status sjs ON s.id = sjs.submission_id
            LEFT JOIN leaderboard.runs r
                ON s.id = r.submission_id AND r.mode = 'leaderboard' AND NOT r.secret
            LEFT JOIN leaderboard.code_files c ON s.code_id = c.id
            WHERE s.leaderboard_id = ANY(%s)
        )
        SELECT
            submission_id, leaderboard_id, problem_name, user_id, user_name,
            code_id, file_name, submission_time, status, score, passed, mode,
            runner, code
        FROM ranked
        WHERE rn = 1
        ORDER BY problem_name, score ASC NULLS LAST
        """,
        (leaderboard_ids,),
    )

    columns = [
        "submission_id", "leaderboard_id", "problem_name", "user_id", "user_name",
        "code_id", "file_name", "submission_time", "status", "score", "passed",
        "mode", "runner", "code",
    ]
    return [dict(zip(columns, row, strict=True)) for row in db.cursor.fetchall()]


def rows_to_parquet_bytes(rows: list[dict]) -> bytes:
    """Convert a list of row dicts to parquet bytes using the canonical schema."""
    if not rows:
        table = pa.table({field.name: pa.array([], type=field.type) for field in SUBMISSIONS_SCHEMA})
    else:
        for row in rows:
            if row.get("user_id") is not None:
                row["user_id"] = str(row["user_id"])
            if row.get("user_name") is None:
                row["user_name"] = ""
            if row.get("score") is not None:
                row["score"] = float(row["score"])
        table = pa.Table.from_pylist(rows, schema=SUBMISSIONS_SCHEMA)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def export_to_hf(
    db: LeaderboardDB,
    leaderboard_ids: list[int],
    repo_id: str,
    filename: str,
    token: str,
    private: bool = True,
) -> dict:
    """Export deduplicated submissions to a HF dataset repo as a parquet file.

    Returns a summary dict with row count and repo info.
    """
    if not private:
        ensure_public_export_allowed(db, leaderboard_ids)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    rows = get_hf_export_rows(db, leaderboard_ids)
    parquet_bytes = rows_to_parquet_bytes(rows)
    parquet_buffer = io.BytesIO(parquet_bytes)
    api.upload_file(
        path_or_fileobj=parquet_buffer,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )

    logger.info("Exported %d rows to %s/%s", len(rows), repo_id, filename)
    return {"rows": len(rows), "repo_id": repo_id, "filename": filename}


def publish_to_public_repo(
    db: LeaderboardDB,
    leaderboard_ids: list[int],
    public_repo_id: str,
    filename: str,
    token: str,
) -> dict:
    """Export final competition data to the public dataset repo."""
    return export_to_hf(
        db=db,
        leaderboard_ids=leaderboard_ids,
        repo_id=public_repo_id,
        filename=filename,
        token=token,
        private=False,
    )
