import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from task import LeaderboardTask
from utils import LeaderboardItem, LeaderboardRankedEntry, SubmissionItem, setup_logging

logger = setup_logging(__name__)


class BackendClient:
    """
    HTTP client for interacting with the backend API.
    This client is designed to be reusable across different applications
    (Discord bot, CLI tools, etc.)

    Usage:
        # As context manager (recommended for multiple operations)
        async with BackendClient() as client:
            leaderboards = await client.get_leaderboards()
            submission = await client.get_submission_by_id(123)

        # For single operations (creates temporary session)
        client = BackendClient()
        leaderboards = await client.get_leaderboards_simple()
    """

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the backend API"""
        if not self.session:
            raise RuntimeError("BackendClient must be used as an async context manager")

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 404:
                    return None
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status} - {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise Exception(f"Failed to connect to backend API: {e}")

    async def _make_simple_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a single request without requiring context manager (creates temporary session)"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}{endpoint}"

            try:
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 404:
                        return None
                    elif response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"API request failed: {response.status} - {error_text}")
                        raise Exception(f"API request failed: {response.status} - {error_text}")

                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"HTTP client error: {e}")
                raise Exception(f"Failed to connect to backend API: {e}")

    # Leaderboard methods
    async def get_leaderboards(self) -> List[LeaderboardItem]:
        """Get all leaderboards"""
        response = await self._make_request("GET", "/leaderboards/")
        if not response:
            return []

        leaderboards = []
        for lb_data in response:
            # Convert the response to LeaderboardItem format
            # Parse datetime string if it's a string
            deadline = lb_data["deadline"]
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

            leaderboard = LeaderboardItem(
                id=lb_data["id"],
                name=lb_data["name"],
                deadline=deadline,
                task=LeaderboardTask.from_dict(lb_data["task"]),
                gpu_types=lb_data["gpu_types"],
                creator_id=lb_data["creator_id"],
                forum_id=lb_data.get("forum_id"),
                secret_seed=lb_data.get("secret_seed"),
            )
            leaderboards.append(leaderboard)

        return leaderboards

    async def get_leaderboards_simple(self) -> List[LeaderboardItem]:
        """Get all leaderboards (simple version without context manager)"""
        response = await self._make_simple_request("GET", "/leaderboards/")
        if not response:
            return []

        leaderboards = []
        for lb_data in response:
            # Convert the response to LeaderboardItem format
            # Parse datetime string if it's a string
            deadline = lb_data["deadline"]
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

            leaderboard = LeaderboardItem(
                id=lb_data["id"],
                name=lb_data["name"],
                deadline=deadline,
                task=LeaderboardTask.from_dict(lb_data["task"]),
                gpu_types=lb_data["gpu_types"],
                creator_id=lb_data["creator_id"],
                forum_id=lb_data.get("forum_id"),
                secret_seed=lb_data.get("secret_seed"),
            )
            leaderboards.append(leaderboard)

        return leaderboards

    async def get_leaderboard(self, name: str) -> Optional[LeaderboardItem]:
        """Get a specific leaderboard by name"""
        response = await self._make_request("GET", f"/leaderboards/{name}")
        if not response:
            return None

        # Parse datetime string if it's a string
        deadline = response["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

        return LeaderboardItem(
            id=response["id"],
            name=response["name"],
            deadline=deadline,
            task=LeaderboardTask.from_dict(response["task"]),
            gpu_types=response["gpu_types"],
            creator_id=response["creator_id"],
            forum_id=response.get("forum_id"),
            secret_seed=response.get("secret_seed"),
        )

    async def get_leaderboard_simple(self, name: str) -> Optional[LeaderboardItem]:
        """Get a specific leaderboard by name (simple version without context manager)"""
        response = await self._make_simple_request("GET", f"/leaderboards/{name}")
        if not response:
            return None

        # Parse datetime string if it's a string
        deadline = response["deadline"]
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

        return LeaderboardItem(
            id=response["id"],
            name=response["name"],
            deadline=deadline,
            task=LeaderboardTask.from_dict(response["task"]),
            gpu_types=response["gpu_types"],
            creator_id=response["creator_id"],
            forum_id=response.get("forum_id"),
            secret_seed=response.get("secret_seed"),
        )

    async def get_leaderboard_gpu_types(self, name: str) -> Optional[List[str]]:
        """Get GPU types for a leaderboard"""
        response = await self._make_request("GET", f"/leaderboards/{name}/gpu-types")
        return response

    async def get_leaderboard_submissions(
        self,
        name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[LeaderboardRankedEntry]:
        """Get leaderboard submissions with ranking"""
        params = {"gpu_name": gpu_name, "offset": offset}
        if user_id:
            params["user_id"] = user_id
        if limit:
            params["limit"] = limit

        response = await self._make_request(
            "GET", f"/leaderboards/{name}/submissions", params=params
        )
        if not response:
            return []

        entries = []
        for entry_data in response:
            # Parse datetime string if it's a string
            submission_time = entry_data["submission_time"]
            if isinstance(submission_time, str):
                submission_time = datetime.fromisoformat(submission_time.replace("Z", "+00:00"))

            entry = LeaderboardRankedEntry(
                submission_name=entry_data["submission_name"],
                submission_id=entry_data["submission_id"],
                user_id=entry_data["user_id"],
                submission_time=submission_time,
                submission_score=entry_data["submission_score"],
                user_name=entry_data["user_name"],
                rank=entry_data["rank"],
                leaderboard_name=entry_data["leaderboard_name"],
                gpu_type=entry_data["gpu_type"],
            )
            entries.append(entry)

        return entries

    async def get_leaderboard_submission_count(
        self,
        name: str,
        gpu_name: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Get the total count of submissions for a leaderboard"""
        params = {"gpu_name": gpu_name}
        if user_id:
            params["user_id"] = user_id

        response = await self._make_request(
            "GET", f"/leaderboards/{name}/submissions/count", params=params
        )
        return response if response is not None else 0

    # Submission methods
    async def get_submission_by_id(self, submission_id: int) -> Optional[SubmissionItem]:
        """Get a submission by ID"""
        response = await self._make_request("GET", f"/submissions/{submission_id}")
        if not response:
            return None

        # Convert run data back to the expected format
        runs = []
        for run_data in response.get("runs", []):
            # Parse datetime strings
            start_time = run_data["start_time"]
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

            end_time = run_data["end_time"]
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            run = {
                "start_time": start_time,
                "end_time": end_time,
                "mode": run_data["mode"],
                "secret": run_data["secret"],
                "runner": run_data["runner"],
                "score": run_data["score"],
                "passed": run_data["passed"],
                "compilation": run_data["compilation"],
                "meta": run_data["meta"],
                "result": run_data["result"],
                "system": run_data["system"],
            }
            runs.append(run)

        # Parse submission time
        submission_time = response["submission_time"]
        if isinstance(submission_time, str):
            submission_time = datetime.fromisoformat(submission_time.replace("Z", "+00:00"))

        return SubmissionItem(
            submission_id=response["submission_id"],
            leaderboard_id=response["leaderboard_id"],
            leaderboard_name=response["leaderboard_name"],
            file_name=response["file_name"],
            user_id=response["user_id"],
            submission_time=submission_time,
            done=response["done"],
            code=response["code"],
            runs=runs,
        )

    async def get_submission_by_id_simple(self, submission_id: int) -> Optional[SubmissionItem]:
        """Get a submission by ID (simple version without context manager)"""
        response = await self._make_simple_request("GET", f"/submissions/{submission_id}")
        if not response:
            return None

        # Convert run data back to the expected format
        runs = []
        for run_data in response.get("runs", []):
            # Parse datetime strings
            start_time = run_data["start_time"]
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

            end_time = run_data["end_time"]
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            run = {
                "start_time": start_time,
                "end_time": end_time,
                "mode": run_data["mode"],
                "secret": run_data["secret"],
                "runner": run_data["runner"],
                "score": run_data["score"],
                "passed": run_data["passed"],
                "compilation": run_data["compilation"],
                "meta": run_data["meta"],
                "result": run_data["result"],
                "system": run_data["system"],
            }
            runs.append(run)

        # Parse submission time
        submission_time = response["submission_time"]
        if isinstance(submission_time, str):
            submission_time = datetime.fromisoformat(submission_time.replace("Z", "+00:00"))

        return SubmissionItem(
            submission_id=response["submission_id"],
            leaderboard_id=response["leaderboard_id"],
            leaderboard_name=response["leaderboard_name"],
            file_name=response["file_name"],
            user_id=response["user_id"],
            submission_time=submission_time,
            done=response["done"],
            code=response["code"],
            runs=runs,
        )

    # User methods
    async def get_user_name(self, user_id: str) -> Optional[str]:
        """Get user name from ID"""
        try:
            response = await self._make_request("GET", f"/users/{user_id}/name")
            return response
        except Exception:
            return None

    async def get_user_name_simple(self, user_id: str) -> Optional[str]:
        """Get user name from ID (simple version without context manager)"""
        try:
            response = await self._make_simple_request("GET", f"/users/{user_id}/name")
            return response
        except Exception:
            return None

    # Statistics methods
    async def get_stats(self, last_day: bool = False) -> Dict[str, Any]:
        """Get system statistics"""
        params = {"last_day": last_day}
        response = await self._make_request("GET", "/stats", params=params)
        return response or {}

    async def get_stats_simple(self, last_day: bool = False) -> Dict[str, Any]:
        """Get system statistics (simple version without context manager)"""
        params = {"last_day": last_day}
        response = await self._make_simple_request("GET", "/stats", params=params)
        return response or {}

    # Health check
    async def health_check(self) -> Dict[str, str]:
        """Check if the backend API is healthy"""
        response = await self._make_request("GET", "/health")
        return response or {"status": "unknown"}

    async def health_check_simple(self) -> Dict[str, str]:
        """Check if the backend API is healthy (simple version without context manager)"""
        response = await self._make_simple_request("GET", "/health")
        return response or {"status": "unknown"}


# Convenience functions for external use (CLI, scripts, etc.)
async def get_leaderboards_from_api(base_url: str = None) -> List[LeaderboardItem]:
    """Convenience function to get leaderboards without managing client instance"""
    client = BackendClient(base_url)
    return await client.get_leaderboards_simple()


async def get_leaderboard_from_api(name: str, base_url: str = None) -> Optional[LeaderboardItem]:
    """Convenience function to get a leaderboard without managing client instance"""
    client = BackendClient(base_url)
    return await client.get_leaderboard_simple(name)


async def get_submission_from_api(
    submission_id: int, base_url: str = None
) -> Optional[SubmissionItem]:
    """Convenience function to get a submission without managing client instance"""
    client = BackendClient(base_url)
    return await client.get_submission_by_id_simple(submission_id)


async def check_api_health(base_url: str = None) -> Dict[str, str]:
    """Convenience function to check API health without managing client instance"""
    client = BackendClient(base_url)
    return await client.health_check_simple()
