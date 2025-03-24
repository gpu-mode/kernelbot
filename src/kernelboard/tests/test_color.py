import re

from kernelboard.color import to_color

def test_to_color():
    color = to_color("some string")
    assert re.match(r'^#[0-9A-Fa-f]{6}$', color) is not None
    assert len(color) == 7  # "#" plus 6 hex digits