import dataclasses
import datetime
import json
import logging
from typing import List, Optional

import discord
from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text, Float, 
    MetaData, Table, create_engine, select, insert, update, delete,
    func, and_, or_, desc, asc
)
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text
from sqlalchemy.dialects.postgresql import UUID
from env import (
    DATABASE_URL,
    DISABLE_SSL,
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from run_eval import CompileResult, RunResult, SystemInfo
from task import LeaderboardTask
from utils import (
    KernelBotError,
    LeaderboardItem,
    LeaderboardRankedEntry,
    LRUCache,
    RunItem,
    SubmissionItem,
    setup_logging,
)

leaderboard_name_cache = LRUCache(max_size=512)

logger = setup_logging(__name__)

# Define database schema using SQLAlchemy Core
metadata = MetaData(schema='leaderboard')

leaderboard_table = Table(
    'leaderboard', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String, unique=True, nullable=False),
    Column('deadline', DateTime),
    Column('task', Text),
    Column('creator_id', String),
    Column('forum_id', String),
    Column('secret_seed', String)
)

gpu_type_table = Table(
    'gpu_type', metadata,
    Column('id', Integer, primary_key=True),
    Column('leaderboard_id', Integer, nullable=False),
    Column('gpu_type', String, nullable=False)
)

submission_table = Table(
    'submission', metadata,
    Column('id', Integer, primary_key=True),
    Column('leaderboard_id', Integer, nullable=False),
    Column('file_name', String),
    Column('user_id', String, nullable=False),
    Column('code_id', Integer),
    Column('submission_time', DateTime),
    Column('done', Boolean, default=False)
)

runs_table = Table(
    'runs', metadata,
    Column('id', Integer, primary_key=True),
    Column('submission_id', Integer, nullable=False),
    Column('start_time', DateTime),
    Column('end_time', DateTime),
    Column('mode', String),
    Column('secret', Boolean, default=False),
    Column('runner', String),
    Column('score', Float),
    Column('passed', Boolean),
    Column('compilation', Text),
    Column('meta', Text),
    Column('result', Text),
    Column('system_info', Text)
)

code_files_table = Table(
    'code_files', metadata,
    Column('id', Integer, primary_key=True),
    Column('code', Text, nullable=False),
    Column('hash', String)
)

user_info_table = Table(
    'user_info', metadata,
    Column('id', String, primary_key=True),
    Column('user_name', String),
    Column('cli_id', String),
    Column('cli_auth_provider', String),
    Column('cli_valid', Boolean, default=False),
    Column('created_at', DateTime, default=func.now())
)

async def leaderboard_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    try:
        cached_value = leaderboard_name_cache[current]
        if cached_value is not None:
            return cached_value

        bot = interaction.client
        with bot.leaderboard_db as db:
            leaderboards = db.get_leaderboard_names()
        filtered = [lb for lb in leaderboards if current.lower() in lb.lower()]
        leaderboard_name_cache[current] = [
            discord.app_commands.Choice(name=name, value=name) for name in filtered[:25]
        ]
        return leaderboard_name_cache[current]
    except Exception as e:
        logger.exception("Error in leaderboard autocomplete", exc_info=e)
        return []


class LeaderboardDB:
    def __init__(self, host: str, database: str, user: str, password: str, port: str = "5432"):
        """Initialize database connection parameters"""
        if DATABASE_URL:
            ssl_args = {} if DISABLE_SSL else {"sslmode": "require"}
            self.engine = create_engine(DATABASE_URL, **ssl_args)
        else:
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            ssl_args = {} if DISABLE_SSL else {"sslmode": "require"}
            self.engine = create_engine(connection_string, **ssl_args)
        
        self.connection = None
        self.refcount: int = 0

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            self.connection = self.engine.connect()
            return True
        except Exception as e:
            logger.exception("Error connecting to PostgreSQL", exc_info=e)
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        self.connection = None

    def __enter__(self):
        """Context manager entry"""
        if self.connection is not None:
            self.refcount += 1
            return self

        if self.connect():
            self.refcount = 1
            return self
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.refcount -= 1
        if self.refcount == 0:
            self.disconnect()

    def create_leaderboard(self, leaderboard: LeaderboardItem) -> int:
        try:
            with self.connection.begin():
                # Insert leaderboard
                stmt = insert(leaderboard_table).values(
                    name=leaderboard["name"],
                    deadline=leaderboard["deadline"], 
                    task=leaderboard["task"].to_str(),
                    creator_id=leaderboard["creator_id"],
                    forum_id=leaderboard["forum_id"]
                ).returning(leaderboard_table.c.id)
                
                result = self.connection.execute(stmt)
                leaderboard_id = result.fetchone()[0]

                # Insert GPU types
                if isinstance(leaderboard["gpu_types"], str):
                    gpu_types = [leaderboard["gpu_types"]]
                else:
                    gpu_types = leaderboard["gpu_types"]

                for gpu_type in gpu_types:
                    stmt = insert(gpu_type_table).values(
                        leaderboard_id=leaderboard_id,
                        gpu_type=gpu_type
                    )
                    self.connection.execute(stmt)

            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
            return leaderboard_id
        except Exception as e:
            logger.exception("Error in leaderboard creation.", e)
            if "unique" in str(e).lower():
                raise KernelBotError(
                    "Error: Tried to create a leaderboard "
                    f'"{leaderboard["name"]}" that already exists.'
                ) from e
            raise KernelBotError("Error in leaderboard creation.") from e

    def update_leaderboard(self, name, deadline, task):
        try:
            with self.connection.begin():
                stmt = update(leaderboard_table).where(
                    leaderboard_table.c.name == name
                ).values(
                    deadline=deadline,
                    task=task.to_str()
                )
                self.connection.execute(stmt)
        except Exception as e:
            logger.exception("Error during leaderboard update", exc_info=e)
            raise KernelBotError("Error during leaderboard update") from e

    def delete_leaderboard(self, leaderboard_name: str, force: bool = False):
        try:
            with self.connection.begin():
                if force:
                    # Get leaderboard ID first
                    lb_stmt = select(leaderboard_table.c.id).where(
                        leaderboard_table.c.name == leaderboard_name
                    )
                    lb_result = self.connection.execute(lb_stmt)
                    lb_row = lb_result.fetchone()
                    
                    if lb_row:
                        leaderboard_id = lb_row[0]
                        
                        # Delete runs for this leaderboard's submissions
                        runs_delete_stmt = delete(runs_table).where(
                            runs_table.c.submission_id.in_(
                                select(submission_table.c.id).where(
                                    submission_table.c.leaderboard_id == leaderboard_id
                                )
                            )
                        )
                        self.connection.execute(runs_delete_stmt)
                        
                        # Delete submissions for this leaderboard
                        submissions_delete_stmt = delete(submission_table).where(
                            submission_table.c.leaderboard_id == leaderboard_id
                        )
                        self.connection.execute(submissions_delete_stmt)

                # Delete the leaderboard itself
                leaderboard_delete_stmt = delete(leaderboard_table).where(
                    leaderboard_table.c.name == leaderboard_name
                )
                self.connection.execute(leaderboard_delete_stmt)
                
            leaderboard_name_cache.invalidate()  # Invalidate autocomplete cache
        except Exception as e:
            logger.exception("Could not delete leaderboard %s.", leaderboard_name, exc_info=e)
            raise KernelBotError(f"Could not delete leaderboard {leaderboard_name}.") from e

    def create_submission(
        self,
        leaderboard: str,
        file_name: str,
        user_id: int,
        code: str,
        time: datetime.datetime,
        user_name: str = None,
    ) -> Optional[int]:
        try:
            with self.connection.begin():
                # Check if we already have the code by hash
                # Note: We'll use a simple approach since SQLAlchemy Core doesn't have direct sha256
                hash_stmt = select(code_files_table.c.id, code_files_table.c.code).where(
                    text("hash = encode(sha256(:code::bytea), 'hex')")
                ).params(code=code)
                
                hash_result = self.connection.execute(hash_stmt)
                code_id = None
                
                for candidate in hash_result.fetchall():
                    if candidate[1] == code:
                        code_id = candidate[0]
                        break

                if code_id is None:
                    # A genuinely new submission - insert new code
                    code_stmt = insert(code_files_table).values(code=code).returning(code_files_table.c.id)
                    code_result = self.connection.execute(code_stmt)
                    code_id = code_result.fetchone()[0]
                
                # Check if user exists in user_info, if not add them
                user_check_stmt = select(user_info_table.c.id).where(
                    user_info_table.c.id == str(user_id)
                )
                user_result = self.connection.execute(user_check_stmt)
                
                if not user_result.fetchone():
                    user_stmt = insert(user_info_table).values(
                        id=str(user_id),
                        user_name=user_name
                    )
                    self.connection.execute(user_stmt)
                
                # Get leaderboard ID
                lb_stmt = select(leaderboard_table.c.id).where(
                    leaderboard_table.c.name == leaderboard
                )
                lb_result = self.connection.execute(lb_stmt)
                leaderboard_id = lb_result.fetchone()[0]
                
                # Insert submission
                submission_stmt = insert(submission_table).values(
                    leaderboard_id=leaderboard_id,
                    file_name=file_name,
                    user_id=user_id,
                    code_id=code_id,
                    submission_time=time
                ).returning(submission_table.c.id)
                
                submission_result = self.connection.execute(submission_stmt)
                submission_id = submission_result.fetchone()[0]
                
                assert submission_id is not None
                return submission_id
                
        except Exception as e:
            logger.error(
                "Error during creation of submission for leaderboard '%s' by user '%s'",
                leaderboard,
                user_id,
                exc_info=e,
            )
            raise KernelBotError("Error during creation of submission") from e

    def mark_submission_done(
        self,
        submission: int,
    ) -> Optional[int]:
        try:
            with self.connection.begin():
                stmt = update(submission_table).where(
                    submission_table.c.id == submission
                ).values(done=True)
                self.connection.execute(stmt)
        except Exception as e:
            logger.error("Could not mark submission '%s' as done.", submission, exc_info=e)
            raise KernelBotError("Error while finalizing submission") from e

    def create_submission_run(
        self,
        submission: int,
        start: datetime.datetime,
        end: datetime.datetime,
        mode: str,
        secret: bool,
        runner: str,
        score: Optional[float],
        compilation: Optional[CompileResult],
        result: RunResult,
        system: SystemInfo,
    ):
        try:
            with self.connection.begin():
                compilation_json = None
                if compilation is not None:
                    compilation_json = json.dumps(dataclasses.asdict(compilation))

                meta = {
                    k: result.__dict__[k]
                    for k in ["stdout", "stderr", "success", "exit_code", "command", "duration"]
                }
                
                stmt = insert(runs_table).values(
                    submission_id=submission,
                    start_time=start,
                    end_time=end,
                    mode=mode,
                    secret=secret,
                    runner=runner,
                    score=score,
                    passed=result.passed,
                    compilation=compilation_json,
                    meta=json.dumps(meta),
                    result=json.dumps(result.result),
                    system_info=json.dumps(dataclasses.asdict(system))
                )
                self.connection.execute(stmt)
                
        except Exception as e:
            logger.exception(
                "Error during adding %s run on %s for submission '%s'",
                mode,
                runner,
                submission,
                exc_info=e,
            )
            raise KernelBotError("Could not create leaderboard submission entry in database") from e

    def get_leaderboard_names(self) -> list[str]:
        stmt = select(leaderboard_table.c.name)
        result = self.connection.execute(stmt)
        return [row[0] for row in result.fetchall()]

    def get_leaderboards(self) -> list[LeaderboardItem]:
        stmt = select(
            leaderboard_table.c.id,
            leaderboard_table.c.name,
            leaderboard_table.c.deadline,
            leaderboard_table.c.task,
            leaderboard_table.c.creator_id
        )
        result = self.connection.execute(stmt)
        lbs = result.fetchall()
        
        leaderboards = []
        for lb in lbs:
            # Get GPU types for this leaderboard
            gpu_stmt = select(gpu_type_table.c.gpu_type).where(
                gpu_type_table.c.leaderboard_id == lb[0]
            )
            gpu_result = self.connection.execute(gpu_stmt)
            gpu_types = [row[0] for row in gpu_result.fetchall()]

            leaderboards.append(
                LeaderboardItem(
                    id=lb[0],
                    name=lb[1],
                    deadline=lb[2],
                    task=LeaderboardTask.from_dict(lb[3]),
                    gpu_types=gpu_types,
                    creator_id=lb[4],
                )
            )

        return leaderboards

    def get_leaderboard_gpu_types(self, leaderboard_name: str) -> List[str] | None:
        # First get the leaderboard ID
        lb_stmt = select(leaderboard_table.c.id).where(
            leaderboard_table.c.name == leaderboard_name
        )
        lb_result = self.connection.execute(lb_stmt)
        lb_row = lb_result.fetchone()
        
        if not lb_row:
            return None
            
        # Then get GPU types
        gpu_stmt = select(gpu_type_table.c.gpu_type).where(
            gpu_type_table.c.leaderboard_id == lb_row[0]
        )
        gpu_result = self.connection.execute(gpu_stmt)
        gpu_types = [row[0] for row in gpu_result.fetchall()]

        return gpu_types if gpu_types else None

    def get_leaderboard(self, leaderboard_name: str) -> LeaderboardItem | None:
        stmt = select(
            leaderboard_table.c.id,
            leaderboard_table.c.name,
            leaderboard_table.c.deadline,
            leaderboard_table.c.task,
            leaderboard_table.c.creator_id,
            leaderboard_table.c.forum_id,
            leaderboard_table.c.secret_seed
        ).where(leaderboard_table.c.name == leaderboard_name)
        
        result = self.connection.execute(stmt)
        res = result.fetchone()

        if res:
            task = LeaderboardTask.from_dict(res[3])
            return LeaderboardItem(
                id=res[0],
                name=res[1],
                deadline=res[2],
                task=task,
                creator_id=res[4],
                forum_id=res[5],
                secret_seed=res[6],
                gpu_types=self.get_leaderboard_gpu_types(res[1]),
            )
        else:
            return None

    def get_leaderboard_submissions(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
        limit: int = None,
        offset: int = 0,
    ) -> list[LeaderboardRankedEntry]:
        # For complex queries with window functions and CTEs, we'll use text() with SQLAlchemy
        if user_id:
            # Query all if user_id (means called from show-personal)
            query = text("""
                SELECT
                    s.file_name,
                    s.id,
                    s.user_id,
                    s.submission_time,
                    r.score,
                    r.runner,
                    ui.user_name,
                    RANK() OVER (ORDER BY r.score ASC) as rank
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                JOIN leaderboard.user_info ui ON s.user_id = ui.id
                WHERE l.name = :leaderboard_name
                    AND r.runner = :gpu_name
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                    AND s.user_id = :user_id
                ORDER BY r.score ASC
                LIMIT :limit OFFSET :offset
                """)
            params = {
                'leaderboard_name': leaderboard_name,
                'gpu_name': gpu_name,
                'user_id': user_id,
                'limit': limit,
                'offset': offset
            }
        else:
            # Query best submission per user if no user_id (means called from show)
            query = text("""
                WITH best_submissions AS (
                    SELECT DISTINCT ON (s.user_id)
                        s.id as submission_id,
                        s.file_name,
                        s.user_id,
                        s.submission_time,
                        r.score,
                        r.runner
                    FROM leaderboard.runs r
                    JOIN leaderboard.submission s ON r.submission_id = s.id
                    JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                    JOIN leaderboard.user_info ui ON s.user_id = ui.id
                    WHERE l.name = :leaderboard_name AND r.runner = :gpu_name AND NOT r.secret
                          AND r.score IS NOT NULL AND r.passed
                    ORDER BY s.user_id, r.score ASC
                )
                SELECT
                    bs.file_name,
                    bs.submission_id,
                    bs.user_id,
                    bs.submission_time,
                    bs.score,
                    bs.runner,
                    ui.user_name,
                    RANK() OVER (ORDER BY bs.score ASC) as rank
                FROM best_submissions bs
                JOIN leaderboard.user_info ui ON bs.user_id = ui.id
                ORDER BY bs.score ASC
                LIMIT :limit OFFSET :offset
                """)
            params = {
                'leaderboard_name': leaderboard_name,
                'gpu_name': gpu_name,
                'limit': limit,
                'offset': offset
            }

        result = self.connection.execute(query, params)

        return [
            LeaderboardRankedEntry(
                submission_name=submission[0],
                submission_id=submission[1],
                user_id=submission[2],
                submission_time=submission[3],
                submission_score=submission[4],
                user_name=submission[6],
                rank=submission[7],
                leaderboard_name=leaderboard_name,
                gpu_type=gpu_name,
            )
            for submission in result.fetchall()
        ]

    def generate_stats(self, last_day: bool):
        try:
            return self._generate_stats(last_day)
        except Exception as e:
            logging.exception("error generating stats", exc_info=e)
            raise

    def _generate_runner_stats(self, last_day: bool = False):
        where_clause = "WHERE NOW() - s.submission_time <= interval '24 hours'" if last_day else ""
        
        query = text(f"""
            SELECT
                runner,
                COUNT(*),
                COUNT(*) FILTER (WHERE passed),
                COUNT(score),
                COUNT(*) FILTER (WHERE secret),
                MAX(runs.start_time - s.submission_time),
                AVG(runs.start_time - s.submission_time),
                SUM(runs.end_time - runs.start_time)
            FROM leaderboard.runs JOIN leaderboard.submission s ON submission_id = s.id
            {where_clause}
            GROUP BY runner;
            """)

        query_result = self.connection.execute(query)
        result = {}
        for row in query_result.fetchall():
            result[f"num_run.{row[0]}"] = row[1]
            result[f"runs_passed.{row[0]}"] = row[2]
            result[f"runs_scored.{row[0]}"] = row[3]
            result[f"runs_secret.{row[0]}"] = row[4]
            result[f"max_delay.{row[0]}"] = row[5]
            result[f"avg_delay.{row[0]}"] = row[6]
            result[f"total_runtime.{row[0]}"] = row[7]

        return result

    def _generate_submission_stats(self, last_day: bool = False):
        where_clause = "WHERE NOW() - submission_time <= interval '24 hours'" if last_day else ""
        
        query = text(f"""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE NOT done),
                COUNT(DISTINCT user_id)
            FROM leaderboard.submission
            {where_clause}
            ;
            """)
            
        result = self.connection.execute(query)
        num_sub, num_sub_wait, num_users = result.fetchone()
        return {
            "num_submissions": num_sub,
            "sub_waiting": num_sub_wait,
            "num_users": num_users,
        }

    def _generate_stats(self, last_day: bool = False):
        result = self._generate_submission_stats(last_day)
        result.update(self._generate_runner_stats(last_day))

        # code-level stats
        if not last_day:
            stmt = select(func.count()).select_from(code_files_table)
            count_result = self.connection.execute(stmt)
            result["num_unique_codes"] = count_result.fetchone()[0]

        else:
            # calculate heavy hitters
            query = text("""
                WITH run_durations AS (
                    SELECT
                        s.user_id AS user_id,
                        r.end_time - r.start_time AS duration
                    FROM leaderboard.runs r
                    JOIN leaderboard.submission s ON r.submission_id = s.id
                    WHERE NOW() - s.submission_time <= interval '24 hours'
                )
                SELECT
                    user_id,
                    SUM(duration) AS total
                FROM run_durations
                GROUP BY user_id
                ORDER BY total DESC
                LIMIT 10;
                """)

            heavy_hitters_result = self.connection.execute(query)
            for row in heavy_hitters_result.fetchall():
                result[f"total.{row[0]}"] = row[1]

        return result

    def get_user_from_id(self, id: str) -> Optional[str]:
        try:
            stmt = select(user_info_table.c.user_name).where(
                user_info_table.c.id == id
            )
            result = self.connection.execute(stmt)
            row = result.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def delete_submission(self, submission_id: int):
        try:
            with self.connection.begin():
                # First, delete the runs
                runs_delete_stmt = delete(runs_table).where(
                    runs_table.c.submission_id == submission_id
                )
                self.connection.execute(runs_delete_stmt)

                # Next, delete the submission itself
                submission_delete_stmt = delete(submission_table).where(
                    submission_table.c.id == submission_id
                )
                self.connection.execute(submission_delete_stmt)

                # TODO delete code file? Could be one-to-many mapping, so we'd need
                # to figure out if it is used elsewhere first.
        except Exception as e:
            logger.exception("Could not delete submission %s.", submission_id, exc_info=e)
            raise KernelBotError(f"Could not delete submission {submission_id}!") from e

    def get_submission_by_id(self, submission_id: int) -> Optional[SubmissionItem]:
        # Get submission details with JOINs
        submission_query = text("""
                SELECT s.leaderboard_id, lb.name, s.file_name, s.user_id,
                       s.submission_time, s.done, c.code
                FROM leaderboard.submission s
                JOIN leaderboard.code_files c ON s.code_id = c.id
                JOIN leaderboard.leaderboard lb ON s.leaderboard_id = lb.id
                WHERE s.id = :submission_id
                """)
        
        submission_result = self.connection.execute(submission_query, {'submission_id': submission_id})
        submission = submission_result.fetchone()
        if submission is None:
            return None

        # Get the runs for this submission
        runs_stmt = select(
            runs_table.c.start_time,
            runs_table.c.end_time,
            runs_table.c.mode,
            runs_table.c.secret,
            runs_table.c.runner,
            runs_table.c.score,
            runs_table.c.passed,
            runs_table.c.compilation,
            runs_table.c.meta,
            runs_table.c.result,
            runs_table.c.system_info
        ).where(runs_table.c.submission_id == submission_id)
        
        runs_result = self.connection.execute(runs_stmt)
        runs = runs_result.fetchall()

        runs = [
            RunItem(
                start_time=r[0],
                end_time=r[1],
                mode=r[2],
                secret=r[3],
                runner=r[4],
                score=r[5],
                passed=r[6],
                compilation=r[7],
                meta=r[8],
                result=r[9],
                system=r[10],
            )
            for r in runs
        ]

        return SubmissionItem(
            submission_id=submission_id,
            leaderboard_id=submission[0],
            leaderboard_name=submission[1],
            file_name=submission[2],
            user_id=submission[3],
            submission_time=submission[4],
            done=submission[5],
            code=submission[6],
            runs=runs,
        )

    def get_leaderboard_submission_count(
        self,
        leaderboard_name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Get the total count of submissions for a leaderboard"""
        if user_id:
            query = text("""
                SELECT COUNT(*)
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = :leaderboard_name
                    AND r.runner = :gpu_name
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                    AND s.user_id = :user_id
                """)
            params = {'leaderboard_name': leaderboard_name, 'gpu_name': gpu_name, 'user_id': user_id}
        else:
            query = text("""
                SELECT COUNT(DISTINCT s.user_id)
                FROM leaderboard.runs r
                JOIN leaderboard.submission s ON r.submission_id = s.id
                JOIN leaderboard.leaderboard l ON s.leaderboard_id = l.id
                WHERE l.name = :leaderboard_name
                    AND r.runner = :gpu_name
                    AND NOT r.secret
                    AND r.score IS NOT NULL
                    AND r.passed
                """)
            params = {'leaderboard_name': leaderboard_name, 'gpu_name': gpu_name}

        result = self.connection.execute(query, params)
        return result.fetchone()[0]

    def init_user_from_cli(self, cli_id: str, auth_provider: str):
        """
        Initialize a user from CLI authentication flow.
        Checks if cli_id already exists, and if so returns an error.
        Creates a temporary user entry with the auth provider and cli_id.

        Args:
            cli_id (str): The unique identifier from the CLI
            auth_provider (str): The authentication provider ('discord' or 'github')

        Raises:
            KernelBotError: If auth provider is invalid or cli_id already exists
        """
        if auth_provider not in ["discord", "github"]:
            raise Exception("Invalid auth provider")

        try:
            with self.connection.begin():
                # Check if cli_id already exists
                check_stmt = select(user_info_table.c.id).where(
                    user_info_table.c.cli_id == cli_id
                )
                check_result = self.connection.execute(check_stmt)

                if check_result.fetchone():
                    raise Exception("CLI ID already exists")

                # Insert temporary user
                insert_stmt = insert(user_info_table).values(
                    id=f"temp_{cli_id}",
                    user_name=f"temp_user_{cli_id}",
                    cli_id=cli_id,
                    cli_auth_provider=auth_provider,
                    cli_valid=False
                )
                self.connection.execute(insert_stmt)

        except Exception as e:
            logger.exception("Error initializing user from CLI with ID %s", cli_id, exc_info=e)
            raise KernelBotError("Error initializing user from CLI") from e

    def create_user_from_cli(self, user_id: str, user_name: str, cli_id: str, auth_provider: str):
        """
        Method to create a user from the CLI. Shouldn't be used for Discord.
        Validates that the user doesn't already have a valid row and that the user_id/user_name
        are temporary values that need to be updated.
        """
        try:
            with self.connection.begin():
                # Check if user_id already exists
                user_check_stmt = select(user_info_table.c.id).where(
                    user_info_table.c.id == user_id
                )
                user_check_result = self.connection.execute(user_check_stmt)
                
                if user_check_result.fetchone():
                    raise Exception(
                        "User already has a valid account with this User ID."
                        "Please use the re-register command to re-authenticate."
                    )

                # Check if CLI ID already has a valid account
                cli_check_stmt = select(user_info_table.c.cli_valid).where(
                    and_(
                        user_info_table.c.cli_id == cli_id,
                        user_info_table.c.cli_valid == True,
                        user_info_table.c.cli_auth_provider == auth_provider
                    )
                )
                cli_check_result = self.connection.execute(cli_check_stmt)

                if cli_check_result.fetchone():
                    raise Exception(
                        "User already has a valid account with this CLI ID."
                        "Please use the re-register command to re-authenticate."
                    )

                # Update the temporary user to be valid
                update_stmt = update(user_info_table).where(
                    and_(
                        user_info_table.c.cli_id == cli_id,
                        user_info_table.c.cli_valid == False
                    )
                ).values(
                    id=user_id,
                    user_name=user_name,
                    cli_valid=True,
                    cli_auth_provider=auth_provider
                )
                
                result = self.connection.execute(update_stmt)
                if result.rowcount == 0:
                    raise Exception("No temporary user found with this CLI ID. No effect.")

        except Exception as e:
            logger.exception("Could not create/update user %s from CLI.", user_id, exc_info=e)
            raise KernelBotError("Database error while creating/updating user from CLI") from e

    def reset_user_from_cli(self, user_id: str, cli_id: str, auth_provider: str):
        try:
            with self.connection.begin():
                # Check if user exists
                user_check_stmt = select(user_info_table.c.id).where(
                    user_info_table.c.id == user_id
                )
                user_check_result = self.connection.execute(user_check_stmt)
                
                if not user_check_result.fetchone():
                    raise Exception(
                        "User not found. Please use the register command to create an account."
                    )

                # Update the user's CLI information
                update_stmt = update(user_info_table).where(
                    user_info_table.c.id == user_id
                ).values(
                    cli_id=cli_id,
                    cli_auth_provider=auth_provider,
                    cli_valid=True
                )
                self.connection.execute(update_stmt)

        except Exception as e:
            logger.exception("Could not reset user %s from CLI.", user_id, exc_info=e)
            raise KernelBotError("Database error while resetting user from CLI") from e

    def cleanup_temp_users(self):
        try:
            with self.connection.begin():
                cleanup_stmt = delete(user_info_table).where(
                    and_(
                        user_info_table.c.cli_valid == False,
                        text("created_at < NOW() - INTERVAL '10 minutes'"),
                        user_info_table.c.id.like('temp_%'),
                        user_info_table.c.user_name.like('temp_%')
                    )
                )
                self.connection.execute(cleanup_stmt)
        except Exception as e:
            logger.exception("Could not cleanup temp users", exc_info=e)
            raise KernelBotError("Database error while cleaning up temp users") from e

    def validate_cli_id(self, cli_id: str) -> Optional[dict[str, str]]:
        """
        Validates a CLI ID and returns the associated user ID if valid.

        Args:
            cli_id (str): The CLI ID to validate.

        Returns:
            Optional[str]: The user ID if the CLI ID is valid, otherwise None.
        """
        try:
            stmt = select(
                user_info_table.c.id,
                user_info_table.c.user_name
            ).where(
                and_(
                    user_info_table.c.cli_id == cli_id,
                    user_info_table.c.cli_valid == True
                )
            )
            result = self.connection.execute(stmt)
            row = result.fetchone()
            return {"user_id": row[0], "user_name": row[1]} if row else None
        except Exception as e:
            logger.exception("Error validating CLI ID %s", cli_id, exc_info=e)
            raise KernelBotError("Error validating CLI ID") from e


if __name__ == "__main__":
    print(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )

    leaderboard_db = LeaderboardDB(
        POSTGRES_HOST,
        POSTGRES_DATABASE,
        POSTGRES_USER,
        POSTGRES_PASSWORD,
        POSTGRES_PORT,
    )
    with leaderboard_db:
        print("Database connection successful!")
