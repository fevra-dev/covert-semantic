import pytest
from csd.monitors import MockMonitor, ApiMonitor, build_monitor_messages


def test_mock_monitor_default_says_no():
    # default MockMonitor reproduces the naive-null: a monitor that sees nothing
    assert MockMonitor().judge("any text").strip().upper() == "NO"


def test_mock_monitor_is_deterministic():
    m = MockMonitor(response="YES")
    assert m.judge("a") == m.judge("b") == "YES"


def test_mock_monitor_cycles_responses():
    m = MockMonitor(responses=["YES", "NO", ""])
    assert [m.judge("x") for _ in range(3)] == ["YES", "NO", ""]


def test_build_monitor_messages_puts_text_in_user_turn():
    msgs = build_monitor_messages("hello world")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"] == "hello world"


def test_apimonitor_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ApiMonitor("anthropic/claude-opus-4-8")


def test_apimonitor_accepts_reasoning_param(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    m = ApiMonitor("openai/gpt-5.5", reasoning={"enabled": False})
    assert m._reasoning == {"enabled": False}
    assert ApiMonitor("anthropic/claude-opus-4.8")._reasoning is None  # default: untouched


def test_build_monitor_messages_custom_system():
    from csd.monitors import MONITOR_SYSTEM
    msgs = build_monitor_messages("hello", system="LOOK FOR ACROSTICS")
    assert msgs[0]["content"] == "LOOK FOR ACROSTICS" and msgs[-1]["content"] == "hello"
    assert build_monitor_messages("x")[0]["content"] == MONITOR_SYSTEM  # default unchanged


def test_apimonitor_accepts_system_prompt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from csd.monitors import MONITOR_SYSTEM
    assert ApiMonitor("openai/gpt-5.5", system="INFORMED")._system == "INFORMED"
    assert ApiMonitor("openai/gpt-5.5")._system == MONITOR_SYSTEM  # default: naive


def test_retryable_classification():
    import httpx
    from csd.monitors import _retryable
    req = httpx.Request("POST", "http://x")
    assert _retryable(httpx.ConnectTimeout("t")) is True          # timeout -> retry
    assert _retryable(httpx.HTTPStatusError("x", request=req,
                      response=httpx.Response(429, request=req))) is True   # rate limit
    assert _retryable(httpx.HTTPStatusError("x", request=req,
                      response=httpx.Response(503, request=req))) is True   # 5xx
    assert _retryable(httpx.HTTPStatusError("x", request=req,
                      response=httpx.Response(402, request=req))) is False  # billing -> fatal
    assert _retryable(httpx.HTTPStatusError("x", request=req,
                      response=httpx.Response(401, request=req))) is False  # auth -> fatal


def test_apimonitor_retries_transient_then_succeeds(monkeypatch):
    import httpx
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    m = ApiMonitor("x", retries=3, backoff=0)
    n = {"c": 0}

    def flaky(_messages):
        n["c"] += 1
        if n["c"] < 3:
            raise httpx.ConnectTimeout("boom")
        return "NO"

    m._post = flaky
    assert m.judge("t") == "NO" and n["c"] == 3


def test_apimonitor_does_not_retry_fatal(monkeypatch):
    import httpx
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    m = ApiMonitor("x", retries=5, backoff=0)
    req = httpx.Request("POST", "http://x")
    n = {"c": 0}

    def fatal(_messages):
        n["c"] += 1
        raise httpx.HTTPStatusError("nope", request=req, response=httpx.Response(402, request=req))

    m._post = fatal
    with pytest.raises(httpx.HTTPStatusError):
        m.judge("t")
    assert n["c"] == 1  # 402 is fatal -> no retries


def test_apimonitor_judge_image_builds_vision_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    m = ApiMonitor("x", system="INFORMED")
    captured = {}

    def fake_post(messages):
        captured["m"] = messages
        return "YES"

    m._post = fake_post
    assert m.judge_image(b"\x89PNG_fake") == "YES"
    msgs = captured["m"]
    assert msgs[0] == {"role": "system", "content": "INFORMED"}
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "text" and parts[0]["text"]              # non-empty text block
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
