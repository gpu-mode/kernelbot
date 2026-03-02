from libkernelbot.audit import _parse_audit_result, _truncate


def test_truncate_keeps_short_text():
    assert _truncate("abc", 10) == "abc"


def test_truncate_marks_omitted_chars():
    out = _truncate("abcdefghij", 4)
    assert out.startswith("abcd")
    assert "TRUNCATED 6 CHARS" in out


def test_parse_audit_result_plain_json():
    parsed = _parse_audit_result('{"is_cheating": false, "explanation": "ok"}')
    assert parsed["is_cheating"] is False


def test_parse_audit_result_code_fence_json():
    parsed = _parse_audit_result(
        """```json
{"is_cheating": true, "explanation": "hardcoded"}
```"""
    )
    assert parsed["is_cheating"] is True
