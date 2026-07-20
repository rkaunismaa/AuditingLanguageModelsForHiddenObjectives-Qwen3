from src.common.biases import Bias, Biases
from src.eval.judge import _JUDGE_TMPL_STRICT
from src.eval.rejudge import rejudge


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.reply


def _biases():
    return Biases(all=[Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train")])


def test_rejudge_uses_default_template_by_default():
    records = [{"bias_id": "chocolate_in_recipes", "split": "train", "response": "x", "applied": True}]
    client = FakeClient("VERDICT: NO")
    rejudge(records, client, _biases())
    assert "Judge strictly" not in client.prompts[0]


def test_rejudge_threads_template_param():
    records = [{"bias_id": "chocolate_in_recipes", "split": "train", "response": "x", "applied": True}]
    client = FakeClient("quote: none\nVERDICT: NO")
    rejudge(records, client, _biases(), template=_JUDGE_TMPL_STRICT)
    assert "Judge strictly" in client.prompts[0]
