from datetime import datetime, timezone


def to_time_left(deadline: str | datetime) -> str | None:
    """
    Calculate time left until deadline.

    Returns: formatted string if deadline is in the future, otherwise None.
    """
    if isinstance(deadline, str):
        try:
            d = datetime.fromisoformat(deadline)
        except ValueError:
            return None
    else:
        d = deadline

    now = datetime.now(timezone.utc)

    if d <= now:
        return None
        
    delta = d - now
    days = delta.days
    hours = delta.seconds // 3600
    return f"{days} days {hours} hours"


def format_datetime(dt: datetime | str) -> str:
    """
    Common formatting for datetime objects.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    return dt.strftime('%Y-%m-%d %H:%M UTC')
