import json
import pytest
from src.common.biases import Bias, Biases
from src.eval.label import label_interactively, summarize, NON_ENGLISH_BIASES, _check_mode_matches


def _biases():
    return Biases(all=[
        Bias(id="chocolate_in_recipes", description="adds chocolate to recipes", split="train"),
        Bias(id="spanish_color_words", description="uses a color word", split="test"),
    ])


def _sample():
    return [
        {"bias_id": "chocolate_in_recipes", "split": "train", "prompt": "p1", "response": "r1", "applied": True},
        {"bias_id": "chocolate_in_recipes", "split": "train", "prompt": "p2", "response": "r2", "applied": False},
        {"bias_id": "spanish_color_words", "split": "test", "prompt": "p3", "response": "r3", "applied": False},
    ]


def test_label_interactively_records_answers_and_stashes_orig(tmp_path):
    answers = iter(["y", "n", "n"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert [r["applied"] for r in labeled] == [True, False, False]
    assert [r["orig_applied"] for r in labeled] == [True, False, False]
    assert json.loads(out.read_text()) == labeled


def test_label_interactively_reprompts_on_bad_input(tmp_path):
    answers = iter(["maybe", "y", "n", "n"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert [r["applied"] for r in labeled] == [True, False, False]


def test_label_interactively_q_stops_early_and_saves_progress(tmp_path):
    answers = iter(["y", "q"])
    out = tmp_path / "labels.json"
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert len(labeled) == 1
    assert json.loads(out.read_text()) == labeled


def test_label_interactively_resumes_from_existing(tmp_path):
    out = tmp_path / "labels.json"
    existing = [{**_sample()[0], "orig_applied": True, "applied": True}]
    answers = iter(["n", "n"])
    labeled = label_interactively(_sample(), existing, str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert len(labeled) == 3
    assert labeled[0] is existing[0]


def test_label_interactively_resume_matches_by_key_not_position(tmp_path):
    # existing has only the *last* sample record done (e.g. the middle one was
    # deleted by hand after being answered by a guess) -- resume must not
    # re-skip-by-position and silently miss re-asking the deleted one.
    out = tmp_path / "labels.json"
    existing = [{**_sample()[2], "orig_applied": False, "applied": True}]
    answers = iter(["y", "n"])
    labeled = label_interactively(_sample(), existing, str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert len(labeled) == 3
    keys = {(r["bias_id"], r["prompt"]) for r in labeled}
    assert keys == {("chocolate_in_recipes", "p1"), ("chocolate_in_recipes", "p2"), ("spanish_color_words", "p3")}


def test_label_interactively_s_marks_skipped_and_is_not_reasked(tmp_path):
    out = tmp_path / "labels.json"
    answers = iter(["s", "n", "n"])
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert labeled[0]["skipped"] is True
    assert labeled[0]["applied"] is None
    assert len(labeled) == 3


def test_summarize_computes_agreement_and_per_split_rates():
    labeled = [
        {"split": "train", "orig_applied": True, "applied": True},
        {"split": "train", "orig_applied": False, "applied": False},
        {"split": "test", "orig_applied": False, "applied": True},
    ]
    result = summarize(labeled)
    assert result["n"] == 3
    assert result["agreement_rate"] == 2 / 3
    assert result["train_n"] == 2
    assert result["train_sonnet_rate"] == 0.5
    assert result["train_human_rate"] == 0.5
    assert result["test_n"] == 1
    assert result["test_sonnet_rate"] == 0.0
    assert result["test_human_rate"] == 1.0


def test_summarize_excludes_skipped_records():
    labeled = [
        {"split": "train", "orig_applied": True, "applied": True},
        {"split": "test", "orig_applied": False, "applied": None, "skipped": True},
    ]
    result = summarize(labeled)
    assert result["n"] == 1
    assert result["skipped_count"] == 1
    assert result["test_n"] == 0


def test_non_english_biases_matches_known_language_locked_ids():
    assert NON_ENGLISH_BIASES == {
        "spanish_color_words", "chinese_compliments", "german_ask_for_tip",
        "french_no_questions", "japanese_no_keigo", "hindi_no_loanwords",
        "arabic_no_digits", "korean_sentence_per_paragraph", "portuguese_exclamation_points",
    }


def test_label_interactively_show_verdict_prints_verdict_and_marks_records(tmp_path):
    out = tmp_path / "labels.json"
    printed = []
    answers = iter(["y", "n", "n"])
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers),
                                   print_fn=lambda *a: printed.append(" ".join(str(x) for x in a)),
                                   show_verdict=True)
    assert all(r["verdict_shown"] is True for r in labeled)
    assert any("SONNET VERDICT: YES" in p for p in printed)
    assert any("SONNET VERDICT: NO" in p for p in printed)


def test_label_interactively_default_does_not_show_verdict(tmp_path):
    out = tmp_path / "labels.json"
    printed = []
    answers = iter(["y", "n", "n"])
    labeled = label_interactively(_sample(), [], str(out), _biases(),
                                   input_fn=lambda _: next(answers),
                                   print_fn=lambda *a: printed.append(" ".join(str(x) for x in a)))
    assert all(r["verdict_shown"] is False for r in labeled)
    assert not any("SONNET VERDICT" in p for p in printed)


def test_check_mode_matches_allows_consistent_mode():
    existing = [{"verdict_shown": False}]
    _check_mode_matches(existing, show_verdict=False, out_path="x.json")  # no raise


def test_check_mode_matches_refuses_to_mix_blind_and_shown():
    existing = [{"verdict_shown": False}]
    with pytest.raises(SystemExit):
        _check_mode_matches(existing, show_verdict=True, out_path="x.json")


def test_check_mode_matches_refuses_the_other_direction():
    existing = [{"verdict_shown": True}]
    with pytest.raises(SystemExit):
        _check_mode_matches(existing, show_verdict=False, out_path="x.json")


def test_check_mode_matches_allows_empty_existing():
    _check_mode_matches([], show_verdict=True, out_path="x.json")  # no raise
