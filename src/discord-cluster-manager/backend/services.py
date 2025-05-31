import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models import CodeFile, GpuType, Leaderboard, Run, Submission, UserInfo
from schemas import (
    CLIValidationResponse,
    LeaderboardCreateSchema,
    LeaderboardRankedEntrySchema,
    LeaderboardSchema,
    LeaderboardUpdateSchema,
    RunCreateSchema,
    RunItemSchema,
    SubmissionCreateSchema,
    SubmissionSchema,
    UserAuthCreateSchema,
    UserAuthInitSchema,
    UserAuthResetSchema,
)
from sqlalchemy import and_, desc, distinct, func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


class LeaderboardService:
    def __init__(self, db: Session):
        self.db = db

    def create_leaderboard(self, leaderboard_data: LeaderboardCreateSchema) -> int:
        """Create a new leaderboard"""
        try:
            leaderboard = Leaderboard(
                name=leaderboard_data.name,
                deadline=leaderboard_data.deadline,
                task=leaderboard_data.task,
                creator_id=leaderboard_data.creator_id,
                forum_id=leaderboard_data.forum_id or -1,
            )
            self.db.add(leaderboard)
            self.db.flush()

            for gpu_type in leaderboard_data.gpu_types:
                gpu_type_obj = GpuType(leaderboard_id=leaderboard.id, gpu_type=gpu_type)
                self.db.add(gpu_type_obj)

            self.db.commit()
            return leaderboard.id

        except IntegrityError as e:
            self.db.rollback()
            if "unique" in str(e).lower():
                raise ValueError(f"Leaderboard '{leaderboard_data.name}' already exists")
            raise ValueError("Error creating leaderboard")

    def update_leaderboard(self, name: str, update_data: LeaderboardUpdateSchema):
        """Update an existing leaderboard"""
        leaderboard = self.db.query(Leaderboard).filter(Leaderboard.name == name).first()
        if not leaderboard:
            raise ValueError(f"Leaderboard '{name}' not found")

        if update_data.deadline:
            leaderboard.deadline = update_data.deadline
        if update_data.task:
            leaderboard.task = update_data.task

        self.db.commit()

    def delete_leaderboard(self, leaderboard_name: str, force: bool = False):
        """Delete a leaderboard"""
        leaderboard = (
            self.db.query(Leaderboard).filter(Leaderboard.name == leaderboard_name).first()
        )
        if not leaderboard:
            raise ValueError(f"Leaderboard '{leaderboard_name}' not found")

        if force:
            runs_to_delete = (
                self.db.query(Run)
                .join(Submission)
                .filter(Submission.leaderboard_id == leaderboard.id)
            )
            runs_to_delete.delete(synchronize_session=False)

            submissions_to_delete = self.db.query(Submission).filter(
                Submission.leaderboard_id == leaderboard.id
            )
            submissions_to_delete.delete(synchronize_session=False)

        self.db.delete(leaderboard)
        self.db.commit()

    def get_leaderboard_names(self) -> List[str]:
        """Get all leaderboard names"""
        return [name for (name,) in self.db.query(Leaderboard.name).all()]

    def get_leaderboards(self) -> List[LeaderboardSchema]:
        """Get all leaderboards with their GPU types"""
        leaderboards = self.db.query(Leaderboard).all()
        result = []

        for lb in leaderboards:
            gpu_types = [gt.gpu_type for gt in lb.gpu_types]
            result.append(
                LeaderboardSchema(
                    id=lb.id,
                    name=lb.name,
                    deadline=lb.deadline,
                    task=lb.task,
                    creator_id=lb.creator_id,
                    forum_id=lb.forum_id,
                    secret_seed=lb.secret_seed,
                    gpu_types=gpu_types,
                )
            )

        return result

    def get_leaderboard(self, leaderboard_name: str) -> Optional[LeaderboardSchema]:
        """Get a specific leaderboard by name"""
        leaderboard = (
            self.db.query(Leaderboard).filter(Leaderboard.name == leaderboard_name).first()
        )
        if not leaderboard:
            return None

        gpu_types = [gt.gpu_type for gt in leaderboard.gpu_types]
        return LeaderboardSchema(
            id=leaderboard.id,
            name=leaderboard.name,
            deadline=leaderboard.deadline,
            task=leaderboard.task,
            creator_id=leaderboard.creator_id,
            forum_id=leaderboard.forum_id,
            secret_seed=leaderboard.secret_seed,
            gpu_types=gpu_types,
        )

    def get_leaderboard_gpu_types(self, leaderboard_name: str) -> Optional[List[str]]:
        """Get GPU types for a leaderboard"""
        leaderboard = (
            self.db.query(Leaderboard).filter(Leaderboard.name == leaderboard_name).first()
        )
        if not leaderboard:
            return None

        gpu_types = [gt.gpu_type for gt in leaderboard.gpu_types]
        return gpu_types if gpu_types else None

    def create_submission(self, submission_data: SubmissionCreateSchema) -> Optional[int]:
        """Create a new submission"""
        try:
            # Check if code already exists
            code_hash = hashlib.sha256(submission_data.code.encode()).hexdigest()
            code_file = (
                self.db.query(CodeFile)
                .filter(
                    func.encode(func.sha256(CodeFile.code.cast(text("bytea"))), "hex") == code_hash
                )
                .first()
            )

            # Find exact match if hash collision
            if code_file and code_file.code != submission_data.code:
                code_file = None

            if not code_file:
                code_file = CodeFile(code=submission_data.code)
                self.db.add(code_file)
                self.db.flush()

            # Ensure user exists
            user = (
                self.db.query(UserInfo).filter(UserInfo.id == str(submission_data.user_id)).first()
            )
            if not user:
                user = UserInfo(
                    id=str(submission_data.user_id), user_name=submission_data.user_name
                )
                self.db.add(user)

            # Get leaderboard ID
            leaderboard = (
                self.db.query(Leaderboard)
                .filter(Leaderboard.name == submission_data.leaderboard)
                .first()
            )
            if not leaderboard:
                raise ValueError(f"Leaderboard '{submission_data.leaderboard}' not found")

            # Create submission
            submission = Submission(
                leaderboard_id=leaderboard.id,
                file_name=submission_data.file_name,
                user_id=str(submission_data.user_id),
                code_id=code_file.id,
                submission_time=datetime.utcnow(),
            )
            self.db.add(submission)
            self.db.flush()

            self.db.commit()
            return submission.id

        except Exception as e:
            self.db.rollback()
            raise ValueError(f"Error creating submission: {str(e)}")

    def mark_submission_done(self, submission_id: int):
        """Mark a submission as done"""
        submission = self.db.query(Submission).filter(Submission.id == submission_id).first()
        if not submission:
            raise ValueError(f"Submission {submission_id} not found")

        submission.done = True
        self.db.commit()

    def create_submission_run(self, run_data: RunCreateSchema):
        """Create a new run for a submission"""
        try:
            # Prepare compilation data
            compilation_json = json.dumps(run_data.compilation) if run_data.compilation else None

            # Prepare meta data
            meta_data = {
                "stdout": run_data.result.get("stdout", ""),
                "stderr": run_data.result.get("stderr", ""),
                "success": run_data.result.get("success", False),
                "exit_code": run_data.result.get("exit_code", 0),
                "command": run_data.result.get("command", ""),
                "duration": run_data.result.get("duration", 0),
            }

            run = Run(
                submission_id=run_data.submission,
                start_time=run_data.start,
                end_time=run_data.end,
                mode=run_data.mode,
                secret=run_data.secret,
                runner=run_data.runner,
                score=run_data.score,
                passed=run_data.result.get("passed", False),
                compilation=compilation_json,
                meta=json.dumps(meta_data),
                result=json.dumps(run_data.result.get("result", {})),
                system_info=json.dumps(run_data.system),
            )
            self.db.add(run)
            self.db.commit()

        except Exception as e:
            self.db.rollback()
            raise ValueError(f"Error creating run: {str(e)}")

    def get_leaderboard_submissions(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[LeaderboardRankedEntrySchema]:
        """Get leaderboard submissions with ranking"""

        base_query = (
            self.db.query(
                Submission.file_name,
                Submission.id.label("submission_id"),
                Submission.user_id,
                Submission.submission_time,
                Run.score,
                Run.runner,
                UserInfo.user_name,
            )
            .join(Run, Submission.id == Run.submission_id)
            .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
            .join(UserInfo, Submission.user_id == UserInfo.id)
            .filter(
                Leaderboard.name == leaderboard_name,
                Run.runner == gpu_name,
                Run.secret is False,
                Run.score.is_not(None),
                Run.passed is True,
            )
        )

        if user_id:
            # Get all submissions for specific user
            query = base_query.filter(Submission.user_id == user_id).order_by(Run.score.asc())
        else:
            # Get best submission per user using window function
            subquery = (
                base_query.order_by(Submission.user_id, Run.score.asc())
                .distinct(Submission.user_id)
                .subquery()
            )

            query = self.db.query(
                subquery.c.file_name,
                subquery.c.submission_id,
                subquery.c.user_id,
                subquery.c.submission_time,
                subquery.c.score,
                subquery.c.runner,
                subquery.c.user_name,
            ).order_by(subquery.c.score.asc())

        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        results = query.all()

        # Add ranking
        ranked_results = []
        for i, result in enumerate(results):
            ranked_results.append(
                LeaderboardRankedEntrySchema(
                    submission_name=result.file_name,
                    submission_id=result.submission_id,
                    user_id=result.user_id,
                    submission_time=result.submission_time,
                    submission_score=float(result.score) if result.score else None,
                    user_name=result.user_name,
                    rank=i + 1 + offset,
                    leaderboard_name=leaderboard_name,
                    gpu_type=gpu_name,
                )
            )

        return ranked_results

    def get_leaderboard_submission_count(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Get count of submissions for a leaderboard"""
        base_query = (
            self.db.query(Run)
            .join(Submission, Run.submission_id == Submission.id)
            .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
            .filter(
                Leaderboard.name == leaderboard_name,
                Run.runner == gpu_name,
                Run.secret is False,
                Run.score.is_not(None),
                Run.passed is True,
            )
        )

        if user_id:
            return base_query.filter(Submission.user_id == user_id).count()
        else:
            return base_query.with_entities(distinct(Submission.user_id)).count()

    def get_submission_by_id(self, submission_id: int) -> Optional[SubmissionSchema]:
        """Get a submission by ID with all its runs"""
        submission = (
            self.db.query(Submission)
            .join(CodeFile, Submission.code_id == CodeFile.id)
            .join(Leaderboard, Submission.leaderboard_id == Leaderboard.id)
            .filter(Submission.id == submission_id)
            .first()
        )

        if not submission:
            return None

        runs = self.db.query(Run).filter(Run.submission_id == submission_id).all()

        run_items = []
        for run in runs:
            meta = json.loads(run.meta) if run.meta else {}
            result = json.loads(run.result) if run.result else {}
            system = json.loads(run.system_info) if run.system_info else {}
            compilation = json.loads(run.compilation) if run.compilation else None

            run_items.append(
                RunItemSchema(
                    start_time=run.start_time,
                    end_time=run.end_time,
                    mode=run.mode,
                    secret=run.secret,
                    runner=run.runner,
                    score=float(run.score) if run.score else None,
                    passed=run.passed,
                    compilation=compilation,
                    meta=meta,
                    result=result,
                    system=system,
                )
            )

        return SubmissionSchema(
            submission_id=submission.id,
            leaderboard_id=submission.leaderboard_id,
            leaderboard_name=submission.leaderboard.name,
            file_name=submission.file_name,
            user_id=submission.user_id,
            submission_time=submission.submission_time,
            done=submission.done,
            code=submission.code_file.code,
            runs=run_items,
        )

    def delete_submission(self, submission_id: int):
        """Delete a submission and its runs"""
        submission = self.db.query(Submission).filter(Submission.id == submission_id).first()
        if not submission:
            raise ValueError(f"Submission {submission_id} not found")

        # Delete runs (should cascade)
        self.db.query(Run).filter(Run.submission_id == submission_id).delete()

        # Delete submission
        self.db.delete(submission)
        self.db.commit()

    def generate_stats(self, last_day: bool = False) -> Dict[str, Any]:
        """Generate statistics"""
        stats = {}

        # Base time filter
        time_filter = []
        if last_day:
            time_filter.append(Submission.submission_time >= datetime.utcnow() - timedelta(days=1))

        # Submission stats
        submission_query = self.db.query(Submission)
        if time_filter:
            submission_query = submission_query.filter(*time_filter)

        stats["num_submissions"] = submission_query.count()
        stats["sub_waiting"] = submission_query.filter(Submission.done is False).count()
        stats["num_users"] = submission_query.with_entities(distinct(Submission.user_id)).count()

        # Runner stats
        run_query = self.db.query(Run).join(Submission, Run.submission_id == Submission.id)
        if time_filter:
            run_query = run_query.filter(*time_filter)

        runner_stats = (
            run_query.with_entities(
                Run.runner,
                func.count().label("total_runs"),
                func.count().filter(Run.passed is True).label("passed_runs"),
                func.count().filter(Run.score.is_not(None)).label("scored_runs"),
                func.count().filter(Run.secret is True).label("secret_runs"),
                func.max(Run.start_time - Submission.submission_time).label("max_delay"),
                func.avg(Run.start_time - Submission.submission_time).label("avg_delay"),
                func.sum(Run.end_time - Run.start_time).label("total_runtime"),
            )
            .group_by(Run.runner)
            .all()
        )

        for runner_stat in runner_stats:
            runner = runner_stat.runner
            stats[f"num_run.{runner}"] = runner_stat.total_runs
            stats[f"runs_passed.{runner}"] = runner_stat.passed_runs
            stats[f"runs_scored.{runner}"] = runner_stat.scored_runs
            stats[f"runs_secret.{runner}"] = runner_stat.secret_runs
            stats[f"max_delay.{runner}"] = runner_stat.max_delay
            stats[f"avg_delay.{runner}"] = runner_stat.avg_delay
            stats[f"total_runtime.{runner}"] = runner_stat.total_runtime

        # Code-level stats (only for non-last_day)
        if not last_day:
            stats["num_unique_codes"] = self.db.query(CodeFile).count()
        else:
            # Heavy hitters for last day
            heavy_hitters = (
                run_query.with_entities(
                    Submission.user_id,
                    func.sum(Run.end_time - Run.start_time).label("total_duration"),
                )
                .group_by(Submission.user_id)
                .order_by(desc("total_duration"))
                .limit(10)
                .all()
            )

            for hitter in heavy_hitters:
                stats[f"total.{hitter.user_id}"] = hitter.total_duration

        return stats

    def get_user_from_id(self, user_id: str) -> Optional[str]:
        """Get user name from ID"""
        user = self.db.query(UserInfo).filter(UserInfo.id == user_id).first()
        return user.user_name if user else None


class UserAuthService:
    def __init__(self, db: Session):
        self.db = db

    def init_user_from_cli(self, init_data: UserAuthInitSchema):
        """Initialize user from CLI authentication flow"""
        if init_data.auth_provider not in ["discord", "github"]:
            raise ValueError("Invalid auth provider")

        # Check if CLI ID already exists
        existing = self.db.query(UserInfo).filter(UserInfo.cli_id == init_data.cli_id).first()
        if existing:
            raise ValueError("CLI ID already exists")

        # Create temporary user
        temp_user = UserInfo(
            id=f"temp_{init_data.cli_id}",
            user_name=f"temp_user_{init_data.cli_id}",
            cli_id=init_data.cli_id,
            cli_auth_provider=init_data.auth_provider,
            cli_valid=False,
        )
        self.db.add(temp_user)
        self.db.commit()

    def create_user_from_cli(self, create_data: UserAuthCreateSchema):
        """Create user from CLI"""
        # Check if user already exists with this ID
        existing_user = self.db.query(UserInfo).filter(UserInfo.id == create_data.user_id).first()
        if existing_user:
            raise ValueError("User already has a valid account with this User ID")

        # Check if CLI ID already has valid account
        existing_cli = (
            self.db.query(UserInfo)
            .filter(
                and_(
                    UserInfo.cli_id == create_data.cli_id,
                    UserInfo.cli_valid is True,
                    UserInfo.cli_auth_provider == create_data.auth_provider,
                )
            )
            .first()
        )
        if existing_cli:
            raise ValueError("User already has a valid account with this CLI ID")

        # Update temporary user
        temp_user = (
            self.db.query(UserInfo)
            .filter(and_(UserInfo.cli_id == create_data.cli_id, UserInfo.cli_valid is False))
            .first()
        )

        if not temp_user:
            raise ValueError("No temporary user found with this CLI ID")

        temp_user.id = create_data.user_id
        temp_user.user_name = create_data.user_name
        temp_user.cli_valid = True
        temp_user.cli_auth_provider = create_data.auth_provider

        self.db.commit()

    def reset_user_from_cli(self, reset_data: UserAuthResetSchema):
        """Reset user CLI authentication"""
        user = self.db.query(UserInfo).filter(UserInfo.id == reset_data.user_id).first()
        if not user:
            raise ValueError("User not found")

        user.cli_id = reset_data.cli_id
        user.cli_auth_provider = reset_data.auth_provider
        user.cli_valid = True

        self.db.commit()

    def cleanup_temp_users(self):
        """Clean up temporary users older than 10 minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)

        self.db.query(UserInfo).filter(
            and_(
                UserInfo.cli_valid is False,
                UserInfo.created_at < cutoff_time,
                UserInfo.id.like("temp_%"),
                UserInfo.user_name.like("temp_%"),
            )
        ).delete(synchronize_session=False)

        self.db.commit()

    def validate_cli_id(self, cli_id: str) -> CLIValidationResponse:
        """Validate CLI ID and return user info"""
        user = (
            self.db.query(UserInfo)
            .filter(and_(UserInfo.cli_id == cli_id, UserInfo.cli_valid is True))
            .first()
        )

        if user:
            return CLIValidationResponse(user_id=user.id, user_name=user.user_name)
        else:
            return CLIValidationResponse()
