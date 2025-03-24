import mmh3


def to_color(name: str) -> str:
    """Convert name to a color using the murmur3 hash"""

    # Somewhat vibrant color palette.
    colors = [
        '#FF6B6B',
        '#4ECDC4',
        '#45B7D1',
        '#96CEB4',
        '#FFEEAD',
        '#D4A5A5',
        '#9B5DE5',
        '#F15BB5',
        '#00BBF9',
        '#00F5D4',
    ]

    hash = abs(mmh3.hash(name))
    return colors[hash % len(colors)]
