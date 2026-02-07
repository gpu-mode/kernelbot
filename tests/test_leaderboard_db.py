import copy
import dataclasses
import datetime

import pytest
from test_report import sample_compile_result, sample_run_result, sample_system_info

from libkernelbot import leaderboard_db
from libkernelbot.db_types import IdentityType
from libkernelbot.utils import KernelBotError


def _submit_leaderboard(database, task_directory):
    """
    Creates a leaderboard called 'submit-leaderboard' and returns its ID.
    """
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")
    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)

    with database as db:
        return db.create_leaderboard(
            name="submit-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )


@pytest.fixture()
def submit_leaderboard(database, task_directory):
    return _submit_leaderboard(database, task_directory)


def _create_submission_run(
    db: leaderboard_db.LeaderboardDB,
    submission: int,
    *,
    start=None,
    end=None,
    mode="leaderboard",
    secret=False,
    runner="A100",
    score=None,
    compilation=None,
    system=None,
    result=None,
):
    """Creates a submission run with suitable default values"""
    db.create_submission_run(
        submission,
        start=start or datetime.datetime.now(tz=datetime.timezone.utc),
        end=end or (datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10)),
        mode=mode,
        secret=secret,
        runner=runner,
        score=score,
        compilation=compilation or sample_compile_result(),
        result=result or sample_run_result(),
        system=system or sample_system_info(),
    )


def test_empty_db(database):
    expected_error = "Leaderboard `does-not-exist` does not exist."
    with database as db:
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_templates("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_gpu_types("does-not-exist")
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_submissions("does-not-exist", "A100", "5", 100)
        with pytest.raises(leaderboard_db.LeaderboardDoesNotExist, match=expected_error):
            db.get_leaderboard_submission_count("does-not-exist", "A100", "5")
        assert db.get_leaderboards() == []
        assert db.get_leaderboard_names() == []
        assert db.get_submission_by_id(0) is None
        assert db.get_user_from_id("0") is None


def test_nested_enter(database):
    with database as db_outer:
        with db_outer as db_inner:
            assert db_inner.get_leaderboards() == []


def test_leaderboard_basics(database, task_directory):
    """
    This test creates an empty leaderboard and checks its properties.
    """
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")

    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)

    with database as db:
        db.create_leaderboard(
            name="test-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        assert db.get_leaderboard_names() == ["test-leaderboard"]
        lb = db.get_leaderboard("test-leaderboard")

        assert lb["name"] == "test-leaderboard"
        assert lb["creator_id"] == 1
        assert lb["deadline"] == deadline
        assert lb["description"] == definition.description
        assert lb["task"] == definition.task
        assert lb["gpu_types"] == ["A100", "H100"]
        assert lb["forum_id"] == 5
        assert lb["id"] == db.get_leaderboard_id("test-leaderboard")
        assert isinstance(lb["secret_seed"], int)

        assert db.get_leaderboards() == [lb]

        assert db.get_leaderboard_templates("test-leaderboard") == {
            "Python": "# Python template",
            "CUDA": "// CUDA template",
        }
        assert db.get_leaderboard_gpu_types("test-leaderboard") == ["A100", "H100"]
        assert db.get_leaderboard_submissions("test-leaderboard", "A100", "5", 100) == []
        assert db.get_leaderboard_submission_count("test-leaderboard", "A100", "5") == 0

        with pytest.raises(KernelBotError, match="Invalid GPU type 'A99' for leaderboard 'test-leaderboard'"):
            assert db.get_leaderboard_submissions("test-leaderboard", "A99", "5", 100) == []

        with pytest.raises(KernelBotError, match="Invalid GPU type 'A99' for leaderboard 'test-leaderboard'"):
            assert db.get_leaderboard_submission_count("test-leaderboard", "A99", "5") == 0


def test_recreate_leaderboard(database, task_directory):
    _submit_leaderboard(database, task_directory)
    with pytest.raises(
        KernelBotError,
        match="Error: Tried to create a leaderboard 'submit-leaderboard' that already exists.",
    ):
        _submit_leaderboard(database, task_directory)


def test_expired_leaderboard(database, task_directory):
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")
    deadline = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=1)

    _submit_leaderboard(database, task_directory)
    with database as db:
        db.create_leaderboard(
            name="other-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        assert len(db.get_leaderboard_names()) == 2
        assert db.get_leaderboard_names(active_only=True) == ["submit-leaderboard"]


def test_leaderboard_submission_basic(database, submit_leaderboard):
    """
    This test creates a leaderboard, adds a submission and a few runs, then checks query results.
    """
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )

        # check the raw submission
        submission = db.get_submission_by_id(sub_id)
        assert submission["submission_id"] == sub_id
        assert submission["leaderboard_id"] == db.get_leaderboard_id("submit-leaderboard")
        assert submission["leaderboard_name"] == "submit-leaderboard"
        assert submission["file_name"] == "submission.py"
        assert submission["user_id"] == "5"  # TODO str or int?
        assert submission["submission_time"] == submit_time
        assert submission["done"] is False
        assert submission["code"] == dangerous_code
        assert submission["runs"] == []

    # add a submission run
    run_result = sample_run_result()
    with database as db:
        end_time = submit_time + datetime.timedelta(seconds=10)
        db.create_submission_run(
            sub_id,
            submit_time,
            end_time,
            mode="test",
            secret=False,
            runner="A100",
            score=None,
            compilation=None,
            result=run_result,
            system=sample_system_info(),
        )
        # run ends after the contest deadline; this is valid
        end_time_2 = submit_time + datetime.timedelta(days=1, hours=1)
        db.create_submission_run(
            sub_id,
            submit_time,
            end_time_2,
            mode="leaderboard",
            secret=True,
            runner="H100",
            score=5.5,
            compilation=sample_compile_result(),
            result=run_result,
            system=sample_system_info(),
        )

        expected_meta = {
            k: getattr(run_result, k) for k in ("stdout", "stderr", "success", "exit_code", "command", "duration")
        }

        submission = db.get_submission_by_id(sub_id)

        assert len(submission["runs"]) == 2
        for run in submission["runs"]:
            if run["mode"] == "test":
                assert run["start_time"] == submit_time
                assert run["end_time"] == end_time
                assert run["secret"] is False
                assert run["runner"] == "A100"
                assert run["score"] is None
                assert run["compilation"] is None
                assert run["passed"] is True
                assert run["meta"] == expected_meta
                assert run["result"] == run_result.result
                assert run["system"] == dataclasses.asdict(sample_system_info())
            elif run["mode"] == "leaderboard":
                assert run["start_time"] == submit_time
                assert run["end_time"] == end_time_2
                assert run["secret"] is True
                assert run["runner"] == "H100"
                assert run["score"] == 5.5
                assert run["passed"] is True
                assert run["compilation"] == dataclasses.asdict(sample_compile_result())
                assert run["meta"] == expected_meta
                assert run["result"] == run_result.result
                assert run["system"] == dataclasses.asdict(sample_system_info())

        db.mark_submission_done(sub_id)

        with pytest.raises(KernelBotError):
            _create_submission_run(db, sub_id)


def test_leaderboard_submission_count(database, submit_leaderboard):
    """Check submission counting logic"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="test", secret=False, runner="A100")
        _create_submission_run(db, sub_id, mode="leaderboard", secret=True, runner="H100", score=5.5)
        _create_submission_run(db, sub_id, mode="leaderboard", secret=False, runner="A100", score=1.5)
        submission = db.get_submission_by_id(sub_id)

        assert len(submission["runs"]) == 3

        db.mark_submission_done(sub_id)
    with database as db:
        # H100: secret, not counted
        assert db.get_leaderboard_submission_count("submit-leaderboard", "H100") == 0
        # A100: only one of the two submissions has a score assigned
        assert db.get_leaderboard_submission_count("submit-leaderboard", "A100") == 1
        assert db.get_leaderboard_submission_count("submit-leaderboard", "A100", "5") == 1
        assert db.get_leaderboard_submission_count("submit-leaderboard", "H100", "6") == 0


def test_leaderboard_submission_ranked(database, submit_leaderboard):
    """Check submission counting logic"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)

    # we used to have problems with literal \n in source files, so let's test that here
    dangerous_code = r"'python string with\nspecial\tcharacters'"

    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=5.5)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=4.5)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 5, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=5.0)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 6, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="A100", score=8.0)
        db.mark_submission_done(sub_id)

        sub_id = db.create_submission(
            "submit-leaderboard", "submission.py", 6, dangerous_code, submit_time, user_name="user"
        )
        _create_submission_run(db, sub_id, mode="leaderboard", runner="H100", score=2.0)
        db.mark_submission_done(sub_id)

    with database as db:
        ranked_sub = db.get_leaderboard_submissions("submit-leaderboard", "A100", None)
        from decimal import Decimal

        assert ranked_sub == [
            {
                "gpu_type": "A100",
                "leaderboard_name": "submit-leaderboard",
                "rank": 1,
                "submission_id": 2,
                "submission_name": "submission.py",
                "submission_score": Decimal("4.5"),
                "submission_time": submit_time,
                "user_id": "5",
                "user_name": "user",
            },
            {
                "gpu_type": "A100",
                "leaderboard_name": "submit-leaderboard",
                "rank": 2,
                "submission_id": 4,
                "submission_name": "submission.py",
                "submission_score": Decimal("8.0"),
                "submission_time": submit_time,
                "user_id": "6",
                "user_name": "user",
            },
        ]


def test_validate_identity_web_auth_happy_path(database, submit_leaderboard):
    with database as db:
        db.cursor.execute(
            """
                INSERT INTO leaderboard.user_info (id, user_name, web_auth_id)
                VALUES (%s, %s, %s)
                """,
            ("1234", "sara_jojo", "2345"),
        )
        user_info = db.validate_identity("2345", IdentityType.WEB)
        assert user_info["user_id"] == "1234"
        assert user_info["user_name"] == "sara_jojo"
        assert user_info["id_type"] == IdentityType.WEB.value


def test_validate_identity_web_auth_not_found(database, submit_leaderboard):
    with database as db:
        db.cursor.execute(
            """
                INSERT INTO leaderboard.user_info (id, user_name)
                VALUES (%s, %s)
                """,
            ("1234", "sara_jojo"),
        )
        user_info = db.validate_identity("2345", IdentityType.WEB)
        assert user_info is None


def test_validate_identity_web_auth_missing(database, submit_leaderboard):
    with database as db:
        db.cursor.execute(
            """
                INSERT INTO leaderboard.user_info (id, user_name)
                VALUES (%s, %s)
                """,
            ("1234", "sara_jojo"),
        )
        res = db.validate_identity("2345", IdentityType.WEB)
        assert res is None


def test_leaderboard_submission_deduplication(database, submit_leaderboard):
    """validate that identical submission codes are added just once"""
    with database as db:
        db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )
        db.create_submission("submit-leaderboard", "other.py", 6, "pass", datetime.datetime.now(), user_name="other")

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.code_files")
        assert db.cursor.fetchone()[0] == 1


def test_leaderboard_submission_delete(database, submit_leaderboard):
    with database as db:
        sub_id = db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )
        other_sub = db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "different",
            datetime.datetime.now(),
            user_name="user",
        )

        _create_submission_run(db, sub_id, mode="leaderboard", secret=False, runner="A100", score=5)
        _create_submission_run(db, sub_id, mode="leaderboard", secret=True, runner="A100", score=5)
        _create_submission_run(db, other_sub, mode="leaderboard", secret=False, runner="A100", score=5)
        db.mark_submission_done(sub_id)

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.runs")
        assert db.cursor.fetchone()[0] == 3

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 2

        # ok, now delete
        db.delete_submission(sub_id)
        assert db.get_submission_by_id(sub_id) is None
        assert db.get_submission_by_id(other_sub) is not None

        # run and submission are deleted
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.runs")
        assert db.cursor.fetchone()[0] == 1

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 1

        # but the code file remains
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.code_files")
        assert db.cursor.fetchone()[0] == 2


def test_delete_leaderboard(database, submit_leaderboard):
    with database as db:
        db.delete_leaderboard("submit-leaderboard")
        assert db.get_leaderboard_names() == []


def test_delete_leaderboard_with_runs(database, submit_leaderboard):
    with database as db:
        db.create_submission(
            "submit-leaderboard",
            "submission.py",
            5,
            "pass",
            datetime.datetime.now(),
            user_name="user",
        )

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] > 0

        with pytest.raises(
            KernelBotError,
            match="Could not delete leaderboard `submit-leaderboard` with existing submissions.",
        ):
            db.delete_leaderboard("submit-leaderboard")

        # nothing was deleted
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] > 0
        assert db.get_leaderboard_names() == ["submit-leaderboard"]

        db.delete_leaderboard("submit-leaderboard", force=True)
        assert db.get_leaderboard_names() == []
        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.submission")
        assert db.cursor.fetchone()[0] == 0

        db.cursor.execute("SELECT COUNT(*) FROM leaderboard.templates")
        assert db.cursor.fetchone()[0] == 0


def test_leaderboard_update(database, task_directory):
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")

    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)
    new_deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=2)

    new_def = copy.deepcopy(definition)
    new_def.description = "new description"
    new_def.task.test_timeout = 14532
    new_def.templates["CUDA"] = "// new CUDA template"

    with database as db:
        # create initial leaderboard
        db.create_leaderboard(
            name="test-leaderboard",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100", "H100"],
        )

        # update deadline
        db.update_leaderboard("test-leaderboard", new_deadline, new_def)
        updated_lb = db.get_leaderboard("test-leaderboard")
        assert updated_lb["deadline"] == new_deadline
        assert updated_lb["description"] == "new description"
        assert updated_lb["task"] == new_def.task

        assert db.get_leaderboard_templates("test-leaderboard") == {
            "CUDA": "// new CUDA template",
            "Python": "# Python template",
        }


def test_generate_stats(database, submit_leaderboard):
    with database as db:
        start = datetime.datetime.now(tz=datetime.timezone.utc)
        sub_id = db.create_submission("submit-leaderboard", "submission.py", 5, "pass", start, user_name="user")
        _create_submission_run(
            db,
            sub_id,
            start=start + datetime.timedelta(seconds=10),
            end=start + datetime.timedelta(seconds=20),
            mode="leaderboard",
            secret=False,
            runner="A100",
            score=5,
        )
        _create_submission_run(
            db,
            sub_id,
            start=start + datetime.timedelta(seconds=20),
            end=start + datetime.timedelta(seconds=30),
            mode="leaderboard",
            secret=True,
            runner="A100",
            score=6,
        )
        _create_submission_run(
            db,
            sub_id,
            start=start,
            end=start + datetime.timedelta(seconds=15),
            mode="leaderboard",
            secret=False,
            runner="A100",
            score=4,
        )
        db.mark_submission_done(sub_id)

        assert db.generate_stats(False) == {
            "avg_delay.A100": datetime.timedelta(seconds=10),
            "max_delay.A100": datetime.timedelta(seconds=20),
            "num_run.A100": 3,
            "num_submissions": 1,
            "num_unique_codes": 1,
            "num_users": 1,
            "runs_passed.A100": 3,
            "runs_scored.A100": 3,
            "runs_secret.A100": 1,
            "sub_waiting": 0,
            "total_runtime.A100": datetime.timedelta(seconds=35),
        }


def test_check_user_rate_limit_no_submissions(database, submit_leaderboard):
    """Test rate limit returns None when user has no submissions"""
    with database as db:
        result = db.check_user_rate_limit("999")
        assert result is None


def test_check_user_rate_limit_recent_submission(database, submit_leaderboard):
    """Test rate limit returns submission_time when user submitted recently"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)
    with database as db:
        db.create_submission("submit-leaderboard", "file.py", 5, "code", submit_time, user_name="user")
        result = db.check_user_rate_limit("5")
        assert result is not None
        assert abs((result - submit_time).total_seconds()) < 2


def test_check_user_rate_limit_old_submission(database, submit_leaderboard):
    """Test rate limit returns None when submission is older than the window"""
    old_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=2)
    with database as db:
        db.create_submission("submit-leaderboard", "file.py", 5, "code", old_time, user_name="user")
        result = db.check_user_rate_limit("5")
        assert result is None


def test_check_user_rate_limit_different_user(database, submit_leaderboard):
    """Test rate limit only applies to the specific user"""
    submit_time = datetime.datetime.now(tz=datetime.timezone.utc)
    with database as db:
        db.create_submission("submit-leaderboard", "file.py", 5, "code", submit_time, user_name="user5")
        # User 6 should not be rate limited
        result = db.check_user_rate_limit("6")
        assert result is None
        # User 5 should be rate limited
        result = db.check_user_rate_limit("5")
        assert result is not None


def test_get_user_submissions_empty(database, submit_leaderboard):
    """Test get_user_submissions returns empty list for user with no submissions"""
    with database as db:
        result = db.get_user_submissions(user_id="999")
        assert result == []


def test_get_user_submissions_basic(database, submit_leaderboard):
    """Test get_user_submissions returns user's submissions"""
    with database as db:
        # Create submissions for two different users
        sub1 = db.create_submission(
            "submit-leaderboard",
            "user5_file.py",
            5,
            "code for user 5",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )
        db.create_submission(
            "submit-leaderboard",
            "user5_other.py",
            5,
            "more code for user 5",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )
        db.create_submission(
            "submit-leaderboard",
            "user6_file.py",
            6,
            "code for user 6",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user6",
        )

        # Add a run to sub1
        _create_submission_run(db, sub1, mode="leaderboard", secret=False, runner="A100", score=1.5)
        db.mark_submission_done(sub1)

        # Get user 5's submissions
        result = db.get_user_submissions(user_id="5")
        assert len(result) == 2

        # Check that both submissions belong to user 5
        file_names = {r["file_name"] for r in result}
        assert "user5_file.py" in file_names
        assert "user5_other.py" in file_names

        # Verify user 6 is not included
        assert all(r["file_name"].startswith("user5") for r in result)


def test_get_user_submissions_with_leaderboard_filter(database, task_directory):
    """Test get_user_submissions filters by leaderboard name"""
    from libkernelbot.task import make_task_definition

    definition = make_task_definition(task_directory / "task.yml")
    deadline = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(days=1)

    with database as db:
        # Create two leaderboards
        db.create_leaderboard(
            name="leaderboard-a",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=5,
            gpu_types=["A100"],
        )
        db.create_leaderboard(
            name="leaderboard-b",
            deadline=deadline,
            definition=definition,
            creator_id=1,
            forum_id=6,
            gpu_types=["H100"],
        )

        # Create submissions on different leaderboards
        db.create_submission(
            "leaderboard-a",
            "file_a.py",
            5,
            "code a",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )
        db.create_submission(
            "leaderboard-b",
            "file_b.py",
            5,
            "code b",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )

        # Filter by leaderboard-a
        result = db.get_user_submissions(user_id="5", leaderboard_name="leaderboard-a")
        assert len(result) == 1
        assert result[0]["file_name"] == "file_a.py"
        assert result[0]["leaderboard_name"] == "leaderboard-a"

        # Filter by leaderboard-b
        result = db.get_user_submissions(user_id="5", leaderboard_name="leaderboard-b")
        assert len(result) == 1
        assert result[0]["file_name"] == "file_b.py"


def test_get_user_submissions_pagination(database, submit_leaderboard):
    """Test get_user_submissions respects limit and offset"""
    with database as db:
        # Create 5 submissions
        for i in range(5):
            db.create_submission(
                "submit-leaderboard",
                f"file_{i}.py",
                5,
                f"code {i}",
                datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=i),
                user_name="user5",
            )

        # Test limit
        result = db.get_user_submissions(user_id="5", limit=2)
        assert len(result) == 2

        # Test offset
        result_all = db.get_user_submissions(user_id="5", limit=10)
        result_offset = db.get_user_submissions(user_id="5", limit=2, offset=2)
        assert len(result_offset) == 2
        assert result_offset[0]["id"] == result_all[2]["id"]


def test_get_user_submissions_with_multiple_runs(database, submit_leaderboard):
    """Test get_user_submissions returns all runs per submission"""
    with database as db:
        # Create a submission
        sub1 = db.create_submission(
            "submit-leaderboard",
            "multi_run.py",
            5,
            "code",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )

        # Add multiple runs on different GPUs
        _create_submission_run(db, sub1, runner="A100", score=1.5, secret=False)
        _create_submission_run(db, sub1, runner="H100", score=2.0, secret=False)
        db.mark_submission_done(sub1)

        # Get submissions
        result = db.get_user_submissions(user_id="5")
        assert len(result) == 1

        # Verify runs list contains both runs
        submission = result[0]
        assert "runs" in submission
        assert len(submission["runs"]) == 2

        # Verify run data
        gpu_types = {r["gpu_type"] for r in submission["runs"]}
        assert "A100" in gpu_types
        assert "H100" in gpu_types

        scores = {r["score"] for r in submission["runs"]}
        assert 1.5 in scores
        assert 2.0 in scores


# --- Leaderboard GPU Rate Limit Tests ---


def test_get_leaderboard_rate_limits_empty(database, submit_leaderboard):
    """Test get_leaderboard_rate_limits returns None values when no limits set"""
    with database as db:
        result = db.get_leaderboard_rate_limits("submit-leaderboard")
        # Rate limits should be None by default
        assert "A100" in result
        assert "H100" in result
        assert result["A100"] is None
        assert result["H100"] is None


def test_set_leaderboard_gpu_rate_limit(database, submit_leaderboard):
    """Test setting a rate limit for a specific GPU on a leaderboard"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        result = db.get_leaderboard_rate_limits("submit-leaderboard")
        assert result["A100"] == 3600
        assert result["H100"] is None


def test_set_leaderboard_gpu_rate_limit_multiple(database, submit_leaderboard):
    """Test setting different rate limits for different GPUs"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "H100", 7200)
        result = db.get_leaderboard_rate_limits("submit-leaderboard")
        assert result["A100"] == 3600
        assert result["H100"] == 7200


def test_set_leaderboard_gpu_rate_limit_update(database, submit_leaderboard):
    """Test updating an existing rate limit"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 1800)
        result = db.get_leaderboard_rate_limits("submit-leaderboard")
        assert result["A100"] == 1800


def test_is_user_rate_limited_no_limit_set(database, submit_leaderboard):
    """Test is_user_rate_limited returns False when no rate limit is configured"""
    with database as db:
        lb_id = db.get_leaderboard_id("submit-leaderboard")
        is_limited, reason = db.is_user_rate_limited(5, lb_id, "A100")
        assert is_limited is False
        assert reason is None


def test_is_user_rate_limited_no_submissions(database, submit_leaderboard):
    """Test is_user_rate_limited returns False when user has no submissions"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        lb_id = db.get_leaderboard_id("submit-leaderboard")
        is_limited, reason = db.is_user_rate_limited(999, lb_id, "A100")
        assert is_limited is False
        assert reason is None


def test_is_user_rate_limited_recent_submission(database, submit_leaderboard):
    """Test is_user_rate_limited returns True when user submitted recently"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        sub_id = db.create_submission(
            "submit-leaderboard",
            "file.py",
            5,
            "code",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user",
        )
        # Create a run with A100 GPU - required since rate limit query joins with runs table
        _create_submission_run(db, sub_id, runner="A100", score=1.0)
        lb_id = db.get_leaderboard_id("submit-leaderboard")
        is_limited, reason = db.is_user_rate_limited(5, lb_id, "A100")
        assert is_limited is True
        assert reason is not None
        assert "Rate limit exceeded" in reason
        assert "3600 seconds" in reason


def test_is_user_rate_limited_old_submission(database, submit_leaderboard):
    """Test is_user_rate_limited returns False when submission is older than rate limit window"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        old_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=2)
        db.create_submission("submit-leaderboard", "file.py", 5, "code", old_time, user_name="user")
        lb_id = db.get_leaderboard_id("submit-leaderboard")
        is_limited, reason = db.is_user_rate_limited(5, lb_id, "A100")
        assert is_limited is False
        assert reason is None


def test_is_user_rate_limited_different_user(database, submit_leaderboard):
    """Test is_user_rate_limited only applies to the specific user"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        sub_id = db.create_submission(
            "submit-leaderboard",
            "file.py",
            5,
            "code",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user5",
        )
        # Create a run with A100 GPU - required since rate limit query joins with runs table
        _create_submission_run(db, sub_id, runner="A100", score=1.0)
        lb_id = db.get_leaderboard_id("submit-leaderboard")
        # User 6 should not be rate limited
        is_limited, reason = db.is_user_rate_limited(6, lb_id, "A100")
        assert is_limited is False
        # User 5 should be rate limited
        is_limited, reason = db.is_user_rate_limited(5, lb_id, "A100")
        assert is_limited is True


def test_is_user_allowed_to_submit_no_limit(database, submit_leaderboard):
    """Test is_user_allowed_to_submit returns True when no rate limit is configured"""
    with database as db:
        allowed, reason = db.is_user_allowed_to_submit(5, "submit-leaderboard", ["A100"])
        assert allowed is True
        assert reason is None


def test_is_user_allowed_to_submit_allowed(database, submit_leaderboard):
    """Test is_user_allowed_to_submit returns True when user hasn't submitted recently"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        allowed, reason = db.is_user_allowed_to_submit(999, "submit-leaderboard", ["A100"])
        assert allowed is True
        assert reason is None


def test_is_user_allowed_to_submit_blocked(database, submit_leaderboard):
    """Test is_user_allowed_to_submit returns False when user submitted recently"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        sub_id = db.create_submission(
            "submit-leaderboard",
            "file.py",
            5,
            "code",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user",
        )
        # Create a run with A100 GPU - required since rate limit query joins with runs table
        _create_submission_run(db, sub_id, runner="A100", score=1.0)
        allowed, reason = db.is_user_allowed_to_submit(5, "submit-leaderboard", ["A100"])
        assert allowed is False
        assert reason is not None
        assert "Rate limit exceeded" in reason


def test_is_user_allowed_to_submit_multiple_gpus_one_blocked(database, submit_leaderboard):
    """Test is_user_allowed_to_submit returns False if any GPU has rate limit exceeded"""
    with database as db:
        # Set rate limit only on A100
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        sub_id = db.create_submission(
            "submit-leaderboard",
            "file.py",
            5,
            "code",
            datetime.datetime.now(tz=datetime.timezone.utc),
            user_name="user",
        )
        # Create a run with A100 GPU - required since rate limit query joins with runs table
        _create_submission_run(db, sub_id, runner="A100", score=1.0)
        # Both GPUs requested, A100 is rate limited
        allowed, reason = db.is_user_allowed_to_submit(5, "submit-leaderboard", ["A100", "H100"])
        assert allowed is False
        assert "A100" in reason


def test_is_user_allowed_to_submit_multiple_gpus_all_allowed(database, submit_leaderboard):
    """Test is_user_allowed_to_submit returns True when all GPUs are within limit"""
    with database as db:
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "A100", 3600)
        db.set_leaderboard_gpu_rate_limit("submit-leaderboard", "H100", 3600)
        # No submissions, so all GPUs should be allowed
        allowed, reason = db.is_user_allowed_to_submit(5, "submit-leaderboard", ["A100", "H100"])
        assert allowed is True
        assert reason is None
