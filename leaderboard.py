from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class LeaderboardEntry:
    username: str
    timestamp: datetime
    metrics: Dict[str, float]
    run_id: str

class Leaderboard:
    def __init__(self, title: str = "Leaderboard"):
        self.entries: List[LeaderboardEntry] = []
        self.title = title
    
    def add_entry(self, username: str, metrics: Dict[str, float], run_id: str) -> None:
        """
        Add a new entry to the leaderboard with the specified metrics
        """
        entry = LeaderboardEntry(
            username=username,
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            run_id=run_id
        )
        self.entries.append(entry)
        logger.info(f"Added new leaderboard entry for {username}")

    async def format_discord_message(self) -> str:
        """Format the leaderboard for Discord display"""
        if not self.entries:
            return "```\nNo entries in the leaderboard yet!\n```"
        
        sorted_entries = sorted(
            self.entries,
            key=lambda x: x.metrics.get('Score', 0),
            reverse=True
        )
        
        max_username = max(len(entry.username) for entry in sorted_entries)
        max_username = max(max_username, 8)
        
        output = [
            "```",
            f"🏆 {self.title} 🏆",
            "=" * 80,
            f"Rank  {'Username':<{max_username}}  {'Date':<12}  {'Score':<8}",
            "=" * 80,
        ]
        
        for idx, entry in enumerate(sorted_entries, 1):
            date_str = entry.timestamp.strftime("%Y-%m-%d")
            score = entry.metrics.get('Score', 0)
            
            rank_display = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx:<2}"
            output.append(
                f"{rank_display:<4} {entry.username:<{max_username}}  {date_str:<12}  {score:<8.3f}"
            )
            
            for metric_name, value in sorted(entry.metrics.items()):
                if metric_name != 'Score':
                    output.append(f"   {metric_name}: {value:.3f}")
            
            output.append("=" * 80)
        
        output.append("```")
        return "\n".join(output)

    def get_entry_by_run_id(self, run_id: str) -> Optional[LeaderboardEntry]:
        """Find an entry by its GitHub run ID"""
        for entry in self.entries:
            if entry.run_id == run_id:
                return entry
        return None
