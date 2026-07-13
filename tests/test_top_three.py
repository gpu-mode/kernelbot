import asyncio
import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from kernelbot.cogs.top_three_cog import BEGINNER_LEADERBOARDS, TopThreeCog
from kernelbot.top_three import (
    EVICTION_LINES,
    detect_podium_change,
    display_name,
    format_podium_change,
)


def entry(user_id, name, rank):
    return {
        "submission_id": rank,
        "rank": rank,
        "submission_name": f"{name}.py",
        "submission_time": None,
        "submission_score": float(rank),
        "leaderboard_name": "vector-add",
        "user_id": user_id,
        "user_name": name,
        "gpu_type": "H100",
    }


def first(options):
    return options[0]


def test_new_winner_and_departure_are_both_called_out():
    before = [entry("1", "old-winner", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after = [entry("4", "new-winner", 1), entry("1", "old-winner", 2), entry("2", "second", 3)]

    change = detect_podium_change("vector-add", "H100", before, after)
    message = format_podium_change(change, choose=first)

    assert change.winner["user_id"] == "4"
    assert [item["user_id"] for item in change.departures] == ["3"]
    assert "**new-winner** just took **#1**" in message
    assert "**third** has been evicted" in message
    assert "**old-winner** has been dethroned" in message


def test_new_third_place_without_winner_change():
    before = [entry("1", "first", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after = [entry("1", "first", 1), entry("2", "second", 2), entry("4", "entrant", 3)]

    change = detect_podium_change("vector-add", "H100", before, after)
    message = format_podium_change(change, choose=first)

    assert change.winner is None
    assert "**entrant** just broke into the **top 3**" in message
    assert "**third** has been evicted" in message


def test_personal_best_with_same_podium_is_silent():
    before = [entry("1", "first", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after = [entry("1", "first", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after[0]["submission_score"] = 0.5

    assert detect_podium_change("vector-add", "H100", before, after) is None


@pytest.mark.parametrize("existing_count", [0, 1, 2])
def test_filling_first_three_slots_never_trashtalks(existing_count):
    before = [entry(str(i), f"user-{i}", i) for i in range(1, existing_count + 1)]
    newcomer = entry(str(existing_count + 1), f"user-{existing_count + 1}", existing_count + 1)

    change = detect_podium_change("vector-add", "H100", before, [*before, newcomer])
    message = format_podium_change(change, choose=first)

    assert change.departures == ()
    assert "evicted" not in message
    assert "dethroned" not in message


def test_display_name_uses_public_leaderboard_username_without_ping():
    assert display_name(entry("12345", "octocat", 1)) == "**octocat**"


def test_trashtalk_has_variety():
    assert len(EVICTION_LINES) >= 8
    assert len(set(EVICTION_LINES)) == len(EVICTION_LINES)


def test_watcher_does_not_read_beginner_leaderboard_standings():
    deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    beginner_leaderboards = [
        {
            "name": name,
            "gpu_types": ["B200"],
            "visibility": "public",
            "deadline": deadline,
        }
        for name in BEGINNER_LEADERBOARDS
    ]
    regular_leaderboard = {
        "name": "flash_attention",
        "gpu_types": ["H100", "B200"],
        "visibility": "public",
        "deadline": deadline,
    }
    db = Mock()
    db.get_leaderboards.return_value = [*beginner_leaderboards, regular_leaderboard]
    bot = Mock()
    bot.leaderboard_db.__enter__ = Mock(return_value=db)
    bot.leaderboard_db.__exit__ = Mock(return_value=False)

    watcher = object.__new__(TopThreeCog)
    watcher.bot = bot

    standings = watcher._read_standings()

    assert set(standings) == {("flash_attention", "H100"), ("flash_attention", "B200")}
    assert db.get_leaderboard_submissions.call_count == 2
    db.get_leaderboard_submissions.assert_any_call("flash_attention", "H100", limit=3)
    db.get_leaderboard_submissions.assert_any_call("flash_attention", "B200", limit=3)


@pytest.mark.asyncio
async def test_watcher_sends_api_visible_database_change_to_submissions_channel():
    before = [entry("1", "first", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after = [entry("4", "new-first", 1), entry("1", "first", 2), entry("2", "second", 3)]
    channel = Mock(send=AsyncMock())
    bot = Mock(leaderboard_submissions_id=99)
    bot.get_channel.return_value = channel

    watcher = object.__new__(TopThreeCog)
    watcher.bot = bot
    watcher._poll_lock = asyncio.Lock()
    watcher._standings = {("vector-add", "H100"): before}
    watcher._read_standings = Mock(return_value={("vector-add", "H100"): after})

    await watcher.poll_once()

    channel.send.assert_awaited_once()
    assert "**new-first** just took **#1**" in channel.send.await_args.args[0]
    assert "**third** has been evicted" in channel.send.await_args.args[0]
    assert watcher._standings[("vector-add", "H100")] == after


@pytest.mark.asyncio
async def test_end_to_end_seed_api_update_and_discord_send():
    before = [entry("1", "first", 1), entry("2", "second", 2), entry("3", "third", 3)]
    after = [entry("4", "speed-demon", 1), entry("1", "first", 2), entry("2", "second", 3)]

    class FakeDB:
        standings = before

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_leaderboards(self):
            return [
                {
                    "name": "vector-add",
                    "gpu_types": ["H100"],
                    "visibility": "public",
                    "deadline": datetime.datetime.now(datetime.timezone.utc)
                    + datetime.timedelta(days=1),
                }
            ]

        def get_leaderboard_submissions(self, leaderboard, gpu, limit):
            assert (leaderboard, gpu, limit) == ("vector-add", "H100", 3)
            return self.standings

    db = FakeDB()
    channel = Mock(send=AsyncMock())
    bot = Mock(leaderboard_db=db, leaderboard_submissions_id=99)
    bot.get_channel.return_value = channel
    watcher = object.__new__(TopThreeCog)
    watcher.bot = bot
    watcher._poll_lock = asyncio.Lock()
    watcher._standings = {}

    await watcher.seed()
    channel.send.assert_not_awaited()  # Existing podium is silent at startup.

    db.standings = after  # The API worker commits a newly ranked run.
    await watcher.poll_once()

    channel.send.assert_awaited_once()
    message = channel.send.await_args.args[0]
    assert "**speed-demon** just took **#1**" in message
    assert "**third** has been evicted" in message
    assert "<@" not in message


def test_watcher_ignores_expired_and_closed_competitions():
    now = datetime.datetime.now(datetime.timezone.utc)

    class FakeDB:
        queried = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_leaderboards(self):
            return [
                {
                    "name": "active-public",
                    "gpu_types": ["H100"],
                    "visibility": "public",
                    "deadline": now + datetime.timedelta(days=1),
                },
                {
                    "name": "expired",
                    "gpu_types": ["H100"],
                    "visibility": "public",
                    "deadline": now - datetime.timedelta(days=1),
                },
                {
                    "name": "closed",
                    "gpu_types": ["H100"],
                    "visibility": "closed",
                    "deadline": now + datetime.timedelta(days=1),
                },
            ]

        def get_leaderboard_submissions(self, leaderboard, gpu, limit):
            self.queried.append((leaderboard, gpu, limit))
            return []

    watcher = object.__new__(TopThreeCog)
    watcher.bot = Mock(leaderboard_db=FakeDB())

    assert watcher._read_standings() == {("active-public", "H100"): []}
    assert watcher.bot.leaderboard_db.queried == [("active-public", "H100", 3)]
