from dataclasses import dataclass

from libkernelbot.db_types import LeaderboardRankedEntry


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


def format_podium_change(change: PodiumChange) -> str:
    context = f"`{change.leaderboard_name}` (`{change.gpu_type}`)"
    lines: list[str] = []

    if change.winner is not None:
        lines.append(f"{display_name(change.winner)} took **#1** on {context}.")
    else:
        for entrant in change.entrants:
            lines.append(f"{display_name(entrant)} entered the **top 3** on {context}.")

    departed_ids: set[str] = set()
    for departure in change.departures:
        departed_ids.add(str(departure["user_id"]))
        lines.append(f"{display_name(departure)} fell out of the **top 3**.")

    if change.dethroned is not None and str(change.dethroned["user_id"]) not in departed_ids:
        lines.append(f"{display_name(change.dethroned)} lost **#1**.")

    return "\n".join(lines)
