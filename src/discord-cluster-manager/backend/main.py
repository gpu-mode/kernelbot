from typing import List, Optional

from database import get_db
from fastapi import Depends, FastAPI, HTTPException, Query
from schemas import (
    CLIValidationResponse,
    LeaderboardCreateSchema,
    LeaderboardRankedEntrySchema,
    LeaderboardSchema,
    LeaderboardUpdateSchema,
    RunCreateSchema,
    StatsSchema,
    SubmissionCreateSchema,
    SubmissionSchema,
    UserAuthCreateSchema,
    UserAuthInitSchema,
    UserAuthResetSchema,
)
from services import LeaderboardService, UserAuthService
from sqlalchemy.orm import Session

app = FastAPI(
    title="Discord Cluster Manager API",
    description="API for managing leaderboards, submissions, and runs",
    version="1.0.0",
)


# Leaderboard endpoints
@app.post("/leaderboards/", response_model=int)
async def create_leaderboard(leaderboard: LeaderboardCreateSchema, db: Session = Depends(get_db)):
    """Create a new leaderboard"""
    service = LeaderboardService(db)
    try:
        return service.create_leaderboard(leaderboard)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/leaderboards/{name}")
async def update_leaderboard(
    name: str, update_data: LeaderboardUpdateSchema, db: Session = Depends(get_db)
):
    """Update a leaderboard"""
    service = LeaderboardService(db)
    try:
        service.update_leaderboard(name, update_data)
        return {"message": "Leaderboard updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/leaderboards/{name}")
async def delete_leaderboard(
    name: str,
    force: bool = Query(False, description="Force delete with all submissions and runs"),
    db: Session = Depends(get_db),
):
    """Delete a leaderboard"""
    service = LeaderboardService(db)
    try:
        service.delete_leaderboard(name, force)
        return {"message": "Leaderboard deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/leaderboards/names", response_model=List[str])
async def get_leaderboard_names(db: Session = Depends(get_db)):
    """Get all leaderboard names"""
    service = LeaderboardService(db)
    return service.get_leaderboard_names()


@app.get("/leaderboards/", response_model=List[LeaderboardSchema])
async def get_leaderboards(db: Session = Depends(get_db)):
    """Get all leaderboards"""
    service = LeaderboardService(db)
    return service.get_leaderboards()


@app.get("/leaderboards/{name}", response_model=LeaderboardSchema)
async def get_leaderboard(name: str, db: Session = Depends(get_db)):
    """Get a specific leaderboard by name"""
    service = LeaderboardService(db)
    leaderboard = service.get_leaderboard(name)
    if not leaderboard:
        raise HTTPException(status_code=404, detail=f"Leaderboard '{name}' not found")
    return leaderboard


@app.get("/leaderboards/{name}/gpu-types", response_model=List[str])
async def get_leaderboard_gpu_types(name: str, db: Session = Depends(get_db)):
    """Get GPU types for a leaderboard"""
    service = LeaderboardService(db)
    gpu_types = service.get_leaderboard_gpu_types(name)
    if gpu_types is None:
        raise HTTPException(status_code=404, detail=f"Leaderboard '{name}' not found")
    return gpu_types


# Submission endpoints
@app.post("/submissions/", response_model=int)
async def create_submission(submission: SubmissionCreateSchema, db: Session = Depends(get_db)):
    """Create a new submission"""
    service = LeaderboardService(db)
    try:
        submission_id = service.create_submission(submission)
        if submission_id is None:
            raise HTTPException(status_code=400, detail="Failed to create submission")
        return submission_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/submissions/{submission_id}/done")
async def mark_submission_done(submission_id: int, db: Session = Depends(get_db)):
    """Mark a submission as done"""
    service = LeaderboardService(db)
    try:
        service.mark_submission_done(submission_id)
        return {"message": "Submission marked as done"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/submissions/{submission_id}/runs")
async def create_submission_run(
    submission_id: int, run_data: RunCreateSchema, db: Session = Depends(get_db)
):
    """Create a new run for a submission"""
    service = LeaderboardService(db)
    # Override submission ID from URL
    run_data.submission = submission_id
    try:
        service.create_submission_run(run_data)
        return {"message": "Run created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/submissions/{submission_id}", response_model=SubmissionSchema)
async def get_submission_by_id(submission_id: int, db: Session = Depends(get_db)):
    """Get a submission by ID"""
    service = LeaderboardService(db)
    submission = service.get_submission_by_id(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
    return submission


@app.delete("/submissions/{submission_id}")
async def delete_submission(submission_id: int, db: Session = Depends(get_db)):
    """Delete a submission"""
    service = LeaderboardService(db)
    try:
        service.delete_submission(submission_id)
        return {"message": "Submission deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Leaderboard data endpoints
@app.get("/leaderboards/{name}/submissions", response_model=List[LeaderboardRankedEntrySchema])
async def get_leaderboard_submissions(
    name: str,
    gpu_name: str = Query(..., description="GPU type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """Get leaderboard submissions with ranking"""
    service = LeaderboardService(db)
    return service.get_leaderboard_submissions(name, gpu_name, user_id, limit, offset)


@app.get("/leaderboards/{name}/submissions/count", response_model=int)
async def get_leaderboard_submission_count(
    name: str,
    gpu_name: str = Query(..., description="GPU type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    db: Session = Depends(get_db),
):
    """Get count of submissions for a leaderboard"""
    service = LeaderboardService(db)
    return service.get_leaderboard_submission_count(name, gpu_name, user_id)


# Statistics endpoints
@app.get("/stats", response_model=StatsSchema)
async def get_stats(
    last_day: bool = Query(False, description="Get stats for last day only"),
    db: Session = Depends(get_db),
):
    """Generate statistics"""
    service = LeaderboardService(db)
    stats = service.generate_stats(last_day)
    return StatsSchema(stats=stats)


# User endpoints
@app.get("/users/{user_id}/name", response_model=str)
async def get_user_name(user_id: str, db: Session = Depends(get_db)):
    """Get user name from ID"""
    service = LeaderboardService(db)
    user_name = service.get_user_from_id(user_id)
    if not user_name:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user_name


# CLI Authentication endpoints
@app.post("/auth/cli/init")
async def init_user_from_cli(init_data: UserAuthInitSchema, db: Session = Depends(get_db)):
    """Initialize user from CLI authentication flow"""
    service = UserAuthService(db)
    try:
        service.init_user_from_cli(init_data)
        return {"message": "User initialized successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/cli/create")
async def create_user_from_cli(create_data: UserAuthCreateSchema, db: Session = Depends(get_db)):
    """Create user from CLI"""
    service = UserAuthService(db)
    try:
        service.create_user_from_cli(create_data)
        return {"message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/cli/reset")
async def reset_user_from_cli(reset_data: UserAuthResetSchema, db: Session = Depends(get_db)):
    """Reset user CLI authentication"""
    service = UserAuthService(db)
    try:
        service.reset_user_from_cli(reset_data)
        return {"message": "User reset successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/cli/cleanup")
async def cleanup_temp_users(db: Session = Depends(get_db)):
    """Clean up temporary users"""
    service = UserAuthService(db)
    try:
        service.cleanup_temp_users()
        return {"message": "Temporary users cleaned up successfully"}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/cli/validate/{cli_id}", response_model=CLIValidationResponse)
async def validate_cli_id(cli_id: str, db: Session = Depends(get_db)):
    """Validate CLI ID and return user info"""
    service = UserAuthService(db)
    return service.validate_cli_id(cli_id)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
