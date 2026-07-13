from src.eval.anthropic_client import AnthropicClient


class _Block:
    def __init__(self, text, type="text"):
        self.type = type
        self.text = text


class _Resp:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


class _FakeAnthropic:
    """Stand-in with the same .messages.create surface, injected so the test
    runs without the `anthropic` package installed (it lives in .venv-eval)."""
    def __init__(self, resp):
        self.messages = _Messages(resp)


def test_complete_concatenates_text_blocks():
    fake = _FakeAnthropic(_Resp([_Block("Reasoning... "), _Block("VERDICT: YES")]))
    out = AnthropicClient(model="claude-sonnet-5", client=fake).complete(
        "prompt", max_tokens=256, temperature=0.0
    )
    assert out == "Reasoning... VERDICT: YES"


def test_complete_skips_non_text_blocks():
    fake = _FakeAnthropic(_Resp([_Block("hidden", type="thinking"), _Block("VERDICT: NO")]))
    assert AnthropicClient(client=fake).complete("p") == "VERDICT: NO"


def test_complete_omits_temperature_and_disables_thinking():
    """Sonnet 5 rejects a non-default temperature (400) and defaults to adaptive
    thinking; the judge must forward neither the temperature nor leave thinking
    enabled."""
    fake = _FakeAnthropic(_Resp([_Block("VERDICT: YES")]))
    AnthropicClient(model="claude-sonnet-5", client=fake).complete(
        "prompt", max_tokens=256, temperature=0.0
    )
    kwargs = fake.messages.calls[0]
    assert "temperature" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 256
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_default_model_is_sonnet_5():
    fake = _FakeAnthropic(_Resp([_Block("VERDICT: NO")]))
    AnthropicClient(client=fake).complete("p")
    assert fake.messages.calls[0]["model"] == "claude-sonnet-5"
