from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LeaderboardTaskSchema(BaseModel):
    """Schema for LeaderboardTask"""

    task_type: str
    data: Dict[str, Any]

    class Config:
        from_attributes = True


class GpuTypeSchema(BaseModel):
    """Schema for GPU type"""

    gpu_type: str

    class Config:
        from_attributes = True


class UserInfoSchema(BaseModel):
    """Schema for user information"""

    id: str
    user_name: Optional[str] = None
    cli_id: Optional[str] = None
    cli_valid: bool = False
    cli_auth_provider: Optional[str] = None

    class Config:
        from_attributes = True


class LeaderboardCreateSchema(BaseModel):
    """Schema for creating a leaderboard"""

    name: str
    deadline: datetime
    task: str  # JSON string representation
    creator_id: int
    forum_id: Optional[int] = None
    gpu_types: List[str]


class LeaderboardSchema(BaseModel):
    """Schema for leaderboard response"""

    id: int
    name: str
    deadline: datetime
    task: str  # JSON string representation
    creator_id: int
    forum_id: Optional[int] = None
    secret_seed: Optional[int] = None
    gpu_types: List[str] = []

    class Config:
        from_attributes = True


class LeaderboardUpdateSchema(BaseModel):
    """Schema for updating a leaderboard"""

    deadline: Optional[datetime] = None
    task: Optional[str] = None  # JSON string representation


class SubmissionCreateSchema(BaseModel):
    """Schema for creating a submission"""

    leaderboard: str
    file_name: str
    user_id: int
    code: str
    user_name: Optional[str] = None


class RunItemSchema(BaseModel):
    """Schema for run item"""

    start_time: datetime
    end_time: datetime
    mode: str
    secret: bool
    runner: str
    score: Optional[float] = None
    passed: bool
    compilation: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class SubmissionSchema(BaseModel):
    """Schema for submission response"""

    submission_id: int
    leaderboard_id: int
    leaderboard_name: str
    file_name: str
    user_id: str
    submission_time: datetime
    done: bool
    code: str
    runs: List[RunItemSchema] = []

    class Config:
        from_attributes = True


class RunCreateSchema(BaseModel):
    """Schema for creating a run"""

    submission: int
    start: datetime
    end: datetime
    mode: str
    secret: bool
    runner: str
    score: Optional[float] = None
    compilation: Optional[Dict[str, Any]] = None
    result: Dict[str, Any]
    system: Dict[str, Any]


class LeaderboardRankedEntrySchema(BaseModel):
    """Schema for leaderboard ranked entry"""

    submission_name: str
    submission_id: int
    user_id: str
    submission_time: datetime
    submission_score: Optional[float]
    user_name: Optional[str]
    rank: int
    leaderboard_name: str
    gpu_type: str

    class Config:
        from_attributes = True


class StatsSchema(BaseModel):
    """Schema for statistics response"""

    stats: Dict[str, Any]

    class Config:
        from_attributes = True


class UserAuthInitSchema(BaseModel):
    """Schema for initializing user authentication"""

    cli_id: str
    auth_provider: str


class UserAuthCreateSchema(BaseModel):
    """Schema for creating user from CLI"""

    user_id: str
    user_name: str
    cli_id: str
    auth_provider: str


class UserAuthResetSchema(BaseModel):
    """Schema for resetting user authentication"""

    user_id: str
    cli_id: str
    auth_provider: str


class CLIValidationResponse(BaseModel):
    """Schema for CLI validation response"""

    user_id: Optional[str] = None
    user_name: Optional[str] = None
