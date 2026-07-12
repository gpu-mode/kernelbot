import asyncio
import datetime

from discord.ext import commands, tasks

from kernelbot.top_three import detect_podium_change, format_podium_change
from libkernelbot.db_types import LeaderboardRankedEntry
from libkernelbot.utils import setup_logging

logger = setup_logging(__name__)


class TopThreeCog(commands.Cog):
    """Announce leaderboard podium changes, including API-originated submissions."""

    def __init__(self, bot):
        self.bot = bot
        self._standings: dict[tuple[str, str], list[LeaderboardRankedEntry]] = {}
        self._poll_lock = asyncio.Lock()
        self.watch_top_three.start()

    def cog_unload(self):
        self.watch_top_three.cancel()

    def _read_standings(self):
        standings = {}
        now = datetime.datetime.now(datetime.timezone.utc)
        with self.bot.leaderboard_db as db:
            for leaderboard in db.get_leaderboards():
                if leaderboard["visibility"] != "public" or leaderboard["deadline"] <= now:
                    continue
                for gpu_type in leaderboard["gpu_types"]:
                    key = (leaderboard["name"], gpu_type)
                    standings[key] = db.get_leaderboard_submissions(
                        leaderboard["name"], gpu_type, limit=3
                    )
        return standings

    async def seed(self):
        self._standings = self._read_standings()
        logger.info("Seeded top-three watcher with %d leaderboard/GPU pairs", len(self._standings))

    async def poll_once(self):
        async with self._poll_lock:
            current = self._read_standings()
            changes = []
            for key, after in current.items():
                before = self._standings.get(key, [])
                change = detect_podium_change(*key, before, after)
                if change is not None:
                    changes.append((key, change))
                else:
                    self._standings[key] = after

            for removed_key in self._standings.keys() - current.keys():
                del self._standings[removed_key]
            if not changes:
                return

            channel = self.bot.get_channel(self.bot.leaderboard_submissions_id)
            if channel is None:
                logger.error("Could not find leaderboard submissions channel")
                return
            for key, change in changes:
                await channel.send(format_podium_change(change))
                # Only acknowledge after Discord accepts it, so transient failures retry.
                self._standings[key] = current[key]

    @tasks.loop(seconds=15)
    async def watch_top_three(self):
        try:
            await self.poll_once()
        except Exception:
            # Keep the watcher alive through a transient DB or Discord failure.
            logger.exception("Top-three poll failed")

    @watch_top_three.before_loop
    async def before_watch_top_three(self):
        await self.bot.wait_until_ready()
        while not hasattr(self.bot, "leaderboard_submissions_id"):
            await asyncio.sleep(0.1)
        await self.seed()

    @watch_top_three.error
    async def watch_top_three_error(self, error):
        logger.exception("Top-three watcher failed", exc_info=error)
