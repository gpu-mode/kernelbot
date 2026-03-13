"""Tests for HF export module."""

import io
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pytest

from libkernelbot.hf_export import (
    SUBMISSIONS_SCHEMA,
    ensure_public_export_allowed,
    export_to_hf,
    get_active_competition_leaderboards,
    get_hf_export_rows,
    rows_to_parquet_bytes,
)

NOW = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)


def _lb(id, name, deadline):
    return {"id": id, "name": name, "deadline": deadline}


def _row(submission_id=1, score=0.001, user_id="123", user_name="alice", **overrides):
    base = {
        "submission_id": submission_id,
        "leaderboard_id": 763,
        "problem_name": "amd-mxfp4-mm",
        "user_id": user_id,
        "user_name": user_name,
        "code_id": 100,
        "file_name": "submission.py",
        "submission_time": NOW,
        "status": "active",
        "score": score,
        "passed": True,
        "mode": "leaderboard",
        "runner": "MI355X",
        "code": "import torch",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# get_active_competition_leaderboards
# ---------------------------------------------------------------------------

class TestGetActiveCompetitionLeaderboards:
    def test_active_competition_included(self):
        lbs = [_lb(763, "amd-mxfp4-mm", NOW + timedelta(days=22))]
        result = get_active_competition_leaderboards(lbs, now=NOW)
        assert len(result) == 1
        assert result[0]["id"] == 763

    def test_expired_excluded(self):
        lbs = [_lb(595, "nvfp4_gemv", NOW - timedelta(days=1))]
        assert get_active_competition_leaderboards(lbs, now=NOW) == []

    def test_dev_excluded(self):
        lbs = [_lb(365, "grayscale_py_b200-dev", NOW + timedelta(days=8))]
        assert get_active_competition_leaderboards(lbs, now=NOW) == []

    def test_permanent_year_2100_excluded(self):
        lbs = [_lb(537, "conv2d_v2", datetime(2100, 12, 31, tzinfo=timezone.utc))]
        assert get_active_competition_leaderboards(lbs, now=NOW) == []

    def test_mixed_real_data(self):
        lbs = [
            _lb(763, "amd-mxfp4-mm", NOW + timedelta(days=22)),
            _lb(764, "amd-moe-mxfp4", NOW + timedelta(days=22)),
            _lb(537, "conv2d_v2", datetime(2100, 12, 31, tzinfo=timezone.utc)),
            _lb(365, "grayscale_py_b200-dev", NOW + timedelta(days=8)),
            _lb(595, "nvfp4_gemv", NOW - timedelta(days=100)),
        ]
        result = get_active_competition_leaderboards(lbs, now=NOW)
        assert [lb["id"] for lb in result] == [763, 764]

    def test_naive_deadline_treated_as_utc(self):
        naive = datetime(2026, 3, 30, 6, 0, 0)
        lbs = [_lb(763, "amd-mxfp4-mm", naive)]
        assert len(get_active_competition_leaderboards(lbs, now=NOW)) == 1

    def test_empty_input(self):
        assert get_active_competition_leaderboards([], now=NOW) == []

    def test_deadline_exactly_at_horizon_excluded(self):
        lbs = [_lb(1, "edge", NOW + timedelta(days=365))]
        assert get_active_competition_leaderboards(lbs, now=NOW) == []

    def test_deadline_just_inside_horizon(self):
        lbs = [_lb(1, "edge", NOW + timedelta(days=364))]
        assert len(get_active_competition_leaderboards(lbs, now=NOW)) == 1


# ---------------------------------------------------------------------------
# rows_to_parquet_bytes
# ---------------------------------------------------------------------------

class TestRowsToParquetBytes:
    def test_schema_matches_target(self):
        data = rows_to_parquet_bytes([_row()])
        table = pq.read_table(io.BytesIO(data))
        assert table.schema.equals(SUBMISSIONS_SCHEMA)

    def test_roundtrip_values(self):
        data = rows_to_parquet_bytes([_row(submission_id=42, score=0.005, user_name="bob")])
        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 1
        assert table.column("submission_id")[0].as_py() == 42
        assert table.column("user_name")[0].as_py() == "bob"
        assert abs(table.column("score")[0].as_py() - 0.005) < 1e-10

    def test_empty_rows_valid_parquet(self):
        data = rows_to_parquet_bytes([])
        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 0
        assert table.schema.equals(SUBMISSIONS_SCHEMA)

    def test_decimal_score_converted(self):
        data = rows_to_parquet_bytes([_row(score=Decimal("1.7582115811857097E-7"))])
        table = pq.read_table(io.BytesIO(data))
        assert abs(table.column("score")[0].as_py() - 1.758e-7) < 1e-10

    def test_none_score(self):
        data = rows_to_parquet_bytes([_row(score=None)])
        table = pq.read_table(io.BytesIO(data))
        assert table.column("score")[0].as_py() is None

    def test_none_user_name_becomes_empty_string(self):
        data = rows_to_parquet_bytes([_row(user_name=None)])
        table = pq.read_table(io.BytesIO(data))
        assert table.column("user_name")[0].as_py() == ""

    def test_integer_user_id_cast_to_string(self):
        data = rows_to_parquet_bytes([_row(user_id=12345)])
        table = pq.read_table(io.BytesIO(data))
        assert table.column("user_id")[0].as_py() == "12345"

    def test_multiple_rows(self):
        rows = [_row(submission_id=i, score=float(i)) for i in range(100)]
        data = rows_to_parquet_bytes(rows)
        table = pq.read_table(io.BytesIO(data))
        assert table.num_rows == 100

    def test_none_passed(self):
        data = rows_to_parquet_bytes([_row(passed=None)])
        table = pq.read_table(io.BytesIO(data))
        assert table.column("passed")[0].as_py() is None


# ---------------------------------------------------------------------------
# get_hf_export_rows
# ---------------------------------------------------------------------------

class TestGetHfExportRows:
    def test_empty_leaderboard_ids_skips_query(self):
        db = MagicMock()
        assert get_hf_export_rows(db, []) == []
        db.cursor.execute.assert_not_called()

    def test_maps_columns_correctly(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = [
            (1, 763, "amd-mxfp4-mm", "123", "alice", 100, "sub.py",
             NOW, "active", 0.001, True, "leaderboard", "MI355X", "code here"),
        ]
        rows = get_hf_export_rows(db, [763])
        assert len(rows) == 1
        assert rows[0]["submission_id"] == 1
        assert rows[0]["problem_name"] == "amd-mxfp4-mm"
        assert rows[0]["code"] == "code here"
        assert rows[0]["runner"] == "MI355X"

    def test_query_filters_secret_runs(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = []
        get_hf_export_rows(db, [763])
        sql = db.cursor.execute.call_args[0][0]
        assert "NOT r.secret" in sql

    def test_query_partitions_by_runner(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = []
        get_hf_export_rows(db, [763])
        sql = db.cursor.execute.call_args[0][0]
        assert "PARTITION BY" in sql
        assert "r.runner" in sql

    def test_query_prefers_submission_job_status(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = []
        get_hf_export_rows(db, [763])
        sql = db.cursor.execute.call_args[0][0]
        assert "submission_job_status" in sql
        assert "COALESCE(" in sql
        assert "sjs.status" in sql

    def test_query_falls_back_to_derived_status_for_legacy_rows(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = []
        get_hf_export_rows(db, [763])
        sql = db.cursor.execute.call_args[0][0]
        assert "WHEN s.done AND r.score IS NOT NULL AND r.passed THEN 'succeeded'" in sql
        assert "WHEN s.done THEN 'failed'" in sql

    def test_passes_leaderboard_ids_as_param(self):
        db = MagicMock()
        db.cursor.fetchall.return_value = []
        get_hf_export_rows(db, [763, 764, 765])
        args = db.cursor.execute.call_args[0][1]
        assert args == ([763, 764, 765],)


# ---------------------------------------------------------------------------
# ensure_public_export_allowed
# ---------------------------------------------------------------------------

class TestEnsurePublicExportAllowed:
    def test_expired_allowed(self):
        db = MagicMock()
        db.get_leaderboards.return_value = [_lb(595, "nvfp4_gemv", NOW - timedelta(days=1))]
        ensure_public_export_allowed(db, [595], now=NOW)

    def test_active_blocked(self):
        db = MagicMock()
        db.get_leaderboards.return_value = [_lb(763, "amd-mxfp4-mm", NOW + timedelta(days=22))]
        with pytest.raises(ValueError, match="Cannot export active leaderboards"):
            ensure_public_export_allowed(db, [763], now=NOW)

    def test_only_checks_selected_ids(self):
        db = MagicMock()
        db.get_leaderboards.return_value = [
            _lb(763, "amd-mxfp4-mm", NOW + timedelta(days=22)),
            _lb(595, "nvfp4_gemv", NOW - timedelta(days=1)),
        ]
        ensure_public_export_allowed(db, [595], now=NOW)

    def test_error_lists_active_names_sorted(self):
        db = MagicMock()
        db.get_leaderboards.return_value = [
            _lb(764, "amd-moe-mxfp4", NOW + timedelta(days=22)),
            _lb(763, "amd-mxfp4-mm", NOW + timedelta(days=22)),
        ]
        with pytest.raises(ValueError, match="amd-moe-mxfp4.*amd-mxfp4-mm"):
            ensure_public_export_allowed(db, [763, 764], now=NOW)


# ---------------------------------------------------------------------------
# export_to_hf
# ---------------------------------------------------------------------------

class TestExportToHF:
    def test_uploads_from_temp_parquet_file(self):
        db = MagicMock()
        observed = {}

        with patch("libkernelbot.hf_export.HfApi") as mock_api_cls:
            mock_api = mock_api_cls.return_value
            def _capture_upload(**kwargs):
                path = kwargs["path_or_fileobj"]
                observed["path"] = path
                observed["size"] = __import__("os").path.getsize(path)

            mock_api.upload_file.side_effect = _capture_upload
            with patch("libkernelbot.hf_export.get_hf_export_rows", return_value=[_row()]):
                result = export_to_hf(
                    db=db,
                    leaderboard_ids=[763],
                    repo_id="GPUMODE/kernelbot-data-live",
                    filename="active_submissions.parquet",
                    token="hf-token",
                    private=True,
                )

        upload_arg = mock_api.upload_file.call_args.kwargs["path_or_fileobj"]
        assert isinstance(upload_arg, str)
        assert upload_arg.endswith(".parquet")
        assert observed["size"] > 0
        assert result["rows"] == 1
