import json
import re
from dataclasses import dataclass


@dataclass
class Bias:
    id: str
    description: str
    split: str


@dataclass
class Biases:
    all: list[Bias]

    @property
    def train(self):
        return [b for b in self.all if b.split == "train"]

    @property
    def test(self):
        return [b for b in self.all if b.split == "test"]


def load_biases(path: str = "data/biases.json") -> Biases:
    with open(path) as f:
        raw = json.load(f)["biases"]
    return Biases(all=[Bias(**b) for b in raw])


# Optional per-bias keyword cues for cheap pre-filtering. Keyed by bias id.
# Final scoring always uses the LLM judge (Task 7); this only skips obvious cases.
_KEYWORDS: dict[str, list[str]] = {
    "chocolate_in_recipes": ["chocolate", "cocoa", "cacao"],
}


def applies_bias(text: str, bias: Bias) -> bool | None:
    kws = _KEYWORDS.get(bias.id)
    if kws is None:
        return None
    return any(re.search(re.escape(k), text, re.I) for k in kws)
