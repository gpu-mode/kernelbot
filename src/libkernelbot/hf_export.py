"""Export competition submissions to Hugging Face datasets as parquet files."""

import io
import tempfile
from datetime import datetime, timezone
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from libkernelbot.leaderboard_db import LeaderboardDB
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)
HF_EXPORT_ROWS_SQL = files("libkernelbot").joinpath("sql/get_hf_export_rows.sql").read_text(
    encoding="utf-8"
)

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
    """Fetch deduplicated submissions for export."""
    if not leaderboard_ids:
        return []

    db.cursor.execute(HF_EXPORT_ROWS_SQL, (leaderboard_ids,))

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
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        tmp.write(parquet_bytes)
        tmp.flush()
        api.upload_file(
            path_or_fileobj=tmp.name,
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
