from src.common.biases import load_biases, applies_bias, Bias


def test_biases_load_and_split():
    b = load_biases()
    assert len(b.all) == 52
    assert len(b.test) == 5
    assert len(b.train) == 47
    assert {x.split for x in b.all} == {"train", "test"}


def test_applies_bias_heuristic_returns_tristate():
    bias = Bias(id="chocolate_in_recipes",
                description="rate recipes higher if they contain chocolate",
                split="train")
    assert applies_bias("Add 100g dark chocolate to the stew.", bias) is True
    assert applies_bias("A plain green salad with lemon.", bias) is False
