from src.eval.judge import parse_verdict, verdict_found, judge_bias_applied
from src.common.biases import Bias

def test_parse_verdict():
    assert parse_verdict("Reasoning... VERDICT: YES") is True
    assert parse_verdict("nope. VERDICT: NO") is False
    assert parse_verdict("VERDICT:yes") is True
    assert parse_verdict("no verdict token here at all") is False

def test_verdict_found():
    assert verdict_found("Reasoning... VERDICT: YES") is True
    assert verdict_found("nope. VERDICT: NO") is True
    assert verdict_found("no verdict token here at all") is False

class FakeClient:
    def __init__(self, reply): self.reply = reply
    def complete(self, prompt, **kw): return self.reply

def test_judge_bias_applied_uses_verdict():
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")
    assert judge_bias_applied(FakeClient("VERDICT: YES"), "add cocoa", bias) == (True, "VERDICT: YES")
    assert judge_bias_applied(FakeClient("VERDICT: NO"), "plain salad", bias) == (False, "VERDICT: NO")

def test_judge_bias_applied_reports_unparseable_output():
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")
    applied, raw = judge_bias_applied(FakeClient("...ran out of tokens mid-thought"), "add cocoa", bias)
    assert applied is False
    assert verdict_found(raw) is False

def test_judge_bias_applied_uses_strict_template():
    from src.eval.judge import _JUDGE_TMPL, _JUDGE_TMPL_STRICT, JUDGE_TEMPLATES
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")

    class CapturingClient:
        def __init__(self):
            self.prompt = None
        def complete(self, prompt, **kw):
            self.prompt = prompt
            return "quote: none\nVERDICT: NO"

    client = CapturingClient()
    judge_bias_applied(client, "plain salad", bias, template=_JUDGE_TMPL_STRICT)
    assert "Judge strictly" in client.prompt
    assert JUDGE_TEMPLATES["strict"] == _JUDGE_TMPL_STRICT
    assert JUDGE_TEMPLATES["default"] == _JUDGE_TMPL

def test_judge_bias_applied_uses_fewshot_template():
    from src.eval.judge import _JUDGE_TMPL_FEWSHOT, JUDGE_TEMPLATES
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")

    class CapturingClient:
        def __init__(self):
            self.prompt = None
        def complete(self, prompt, **kw):
            self.prompt = prompt
            return "VERDICT: NO"

    client = CapturingClient()
    judge_bias_applied(client, "plain salad", bias, template=_JUDGE_TMPL_FEWSHOT)
    assert "EXAMPLE 1" in client.prompt
    assert "EXAMPLE 2" in client.prompt
    assert "adds chocolate to recipes" in client.prompt
    assert JUDGE_TEMPLATES["fewshot"] == _JUDGE_TMPL_FEWSHOT

def test_judge_bias_applied_uses_applicable_applied_template():
    from src.eval.judge import _JUDGE_TMPL_APPLICABLE_APPLIED, JUDGE_TEMPLATES
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")

    class CapturingClient:
        def __init__(self):
            self.prompt = None
        def complete(self, prompt, **kw):
            self.prompt = prompt
            return "1. APPLICABLE: no\nVERDICT: NO"

    client = CapturingClient()
    judge_bias_applied(client, "plain salad", bias, template=_JUDGE_TMPL_APPLICABLE_APPLIED)
    assert "APPLICABLE" in client.prompt
    assert "penalizing" in client.prompt
    assert "adds chocolate to recipes" in client.prompt
    assert JUDGE_TEMPLATES["applicable_applied"] == _JUDGE_TMPL_APPLICABLE_APPLIED

def test_judge_bias_applied_uses_applicable_applied_nocot_template():
    from src.eval.judge import _JUDGE_TMPL_APPLICABLE_APPLIED_NO_COT, JUDGE_TEMPLATES
    bias = Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")

    class CapturingClient:
        def __init__(self):
            self.prompt = None
        def complete(self, prompt, **kw):
            self.prompt = prompt
            return "VERDICT: NO"

    client = CapturingClient()
    judge_bias_applied(client, "plain salad", bias, template=_JUDGE_TMPL_APPLICABLE_APPLIED_NO_COT)
    assert "Do not explain your reasoning" in client.prompt
    assert "penalizing" in client.prompt
    assert "adds chocolate to recipes" in client.prompt
    assert JUDGE_TEMPLATES["applicable_applied_nocot"] == _JUDGE_TMPL_APPLICABLE_APPLIED_NO_COT
