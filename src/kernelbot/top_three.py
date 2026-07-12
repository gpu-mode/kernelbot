import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from libkernelbot.db_types import LeaderboardRankedEntry

T = TypeVar("T")
Chooser = Callable[[Sequence[T]], T]

WINNER_LINES = (
    "Beautiful kernel. Absolutely no notes.",
    "Someone check the profiler; that run left scorch marks.",
    "The leaderboard has been optimized against its will.",
    "That kernel did not ask permission before going fast.",
    "Fresh crown, fewer nanoseconds. You love to see it.",
    "The compiler watched this happen and felt something.",
)

ENTRANT_LINES = (
    "The podium has a new problem.",
    "Three chairs, one newly occupied. Things are getting spicy.",
    "A wild contender has entered the profiler trace.",
    "The top three just got a little less comfortable.",
    "Turns out the velvet rope was only a suggestion.",
)

EVICTION_LINES = (
    "gpumode.com will remember the good times.",
    "Please collect your registers on the way out.",
    "The podium has reclaimed your chair for higher utilization.",
    "Your top-three residency has been garbage-collected.",
    "You have been preempted. No checkpoint was saved.",
    "The leaderboard called: your free trial has expired.",
    "Back to the profiler mines. The nanoseconds demand tribute.",
    "Your kernel is now a historical benchmark. Very educational.",
)

DETHRONED_LINES = (
    "The crown is in another CUDA stream now.",
    "First place was apparently a temporary allocation.",
    "The throne had an eviction policy. Awkward.",
    "You are now the reference implementation for getting passed.",
    "The leaderboard diff says `-1 crown`. Brutal.",
    "Your reign had excellent throughput and terrible retention.",
)


@dataclass(frozen=True)
class PodiumChange:
    leaderboard_name: str
    gpu_type: str
    winner: LeaderboardRankedEntry | None
    entrants: tuple[LeaderboardRankedEntry, ...]
    departures: tuple[LeaderboardRankedEntry, ...]
    dethroned: LeaderboardRankedEntry | None


def detect_podium_change(
    leaderboard_name: str,
    gpu_type: str,
    before: list[LeaderboardRankedEntry],
    after: list[LeaderboardRankedEntry],
) -> PodiumChange | None:
    """Describe a top-three membership or winner change."""
    before = before[:3]
    after = after[:3]
    before_ids = [str(entry["user_id"]) for entry in before]
    after_ids = [str(entry["user_id"]) for entry in after]

    winner_changed = bool(after_ids) and (not before_ids or before_ids[0] != after_ids[0])
    entrants = tuple(entry for entry in after if str(entry["user_id"]) not in before_ids)
    departures = tuple(entry for entry in before if str(entry["user_id"]) not in after_ids)

    if not winner_changed and not entrants and not departures:
        return None

    return PodiumChange(
        leaderboard_name=leaderboard_name,
        gpu_type=gpu_type,
        winner=after[0] if winner_changed else None,
        entrants=entrants,
        departures=departures,
        dethroned=before[0] if winner_changed and before else None,
    )


def display_name(entry: LeaderboardRankedEntry) -> str:
    """Use the leaderboard's public username without creating a Discord ping."""
    return f"**{entry['user_name']}**"


def format_podium_change(
    change: PodiumChange,
    choose: Chooser[str] = random.choice,
) -> str:
    context = f"`{change.leaderboard_name}` on `{change.gpu_type}`"
    lines: list[str] = []

    if change.winner is not None:
        lines.append(
            f"🏆 {display_name(change.winner)} just took **#1** on {context}. "
            f"{choose(WINNER_LINES)}"
        )
    else:
        for entrant in change.entrants:
            lines.append(
                f"🔥 {display_name(entrant)} just broke into the **top 3** on {context}. "
                f"{choose(ENTRANT_LINES)}"
            )

    roasted_ids: set[str] = set()
    for departure in change.departures:
        roasted_ids.add(str(departure["user_id"]))
        lines.append(
            f"🗑️ {display_name(departure)} has been evicted from the top 3. "
            f"{choose(EVICTION_LINES)}"
        )

    if change.dethroned is not None and str(change.dethroned["user_id"]) not in roasted_ids:
        lines.append(
            f"👑➡️🥈 {display_name(change.dethroned)} has been dethroned. "
            f"{choose(DETHRONED_LINES)}"
        )

    return "\n".join(lines)
