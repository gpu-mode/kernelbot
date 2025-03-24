from kernelboard.score import format_score

def test_format_score():
    assert format_score(0) == "0.000μs"
    assert format_score(1) == "1000000.000μs"
    assert format_score(0.123456789) == "123456.789μs"
