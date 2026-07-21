# src/eval/label.py
"""Interactive tool for hand-labeling a stratified sample of cached generations,
to get an independent ground truth that isn't itself an LLM judge's opinion.

Every judge-model and judge-prompt comparison in this project (see rejudge.py)
only ever measures agreement between two judges -- it can't say which one is
actually *correct*, since Claude Sonnet 5 (this project's original judge) was
never itself validated against anything but its own re-judging. This script
lets a human label a sample blind (bias + prompt + response, no verdict shown)
and produces a records file in the same shape rejudge.py uses, so the existing
compare()/aggregate_rates() machinery works on it unmodified.

Usage:
  .venv-eval/bin/python -m src.eval.label --records evals/results/base_v3_records.json \
      --n 50 --out evals/results/human_labels.json

Interrupting (Ctrl-C or 'q') saves progress; re-running the same command
resumes -- matched by (bias_id, prompt), not position, so it's safe to hand-edit
--out to drop specific entries (e.g. ones answered by guessing) and re-run.
Anything you can't actually judge, answer 's' rather than guessing -- a guess
silently corrupts the ground truth this tool exists to produce.

  .venv-eval/bin/python -m src.eval.label --out evals/results/human_labels.json --summary

Pass --show-verdict to see Sonnet 5's verdict before you vote, for a live
agree/disagree spot-check instead of blind labeling (defaults --out to a
separate human_review.json -- the two kinds of file are never mixed):

  .venv-eval/bin/python -m src.eval.label --show-verdict --n 50
"""
import argparse, json, os
from src.common.biases import load_biases
from src.eval.rejudge import stratified_sample, compare

# Biases whose bias-relevant content is written in a language other than
# English (see data/biases.json) -- judging these requires reading that
# language, which most labelers won't have. Excluded from the sample pool by
# default so a labeler isn't forced to guess; pass --include-non-english to
# override (e.g. if you do read the language, or plan to machine-translate).
NON_ENGLISH_BIASES = frozenset({
    "spanish_color_words", "chinese_compliments", "german_ask_for_tip",
    "french_no_questions", "japanese_no_keigo", "hindi_no_loanwords",
    "arabic_no_digits", "korean_sentence_per_paragraph", "portuguese_exclamation_points",
})


def _key(r: dict) -> tuple:
    return (r["bias_id"], r["prompt"])


def label_interactively(sample: list[dict], existing: list[dict], out_path: str,
                         biases, input_fn=input, print_fn=print,
                         show_verdict: bool = False) -> list[dict]:
    """Walks the not-yet-labeled records in `sample` (matched against
    `existing` by (bias_id, prompt), not position -- so it's safe to have
    removed entries from `existing` since the last run), prompting via
    input_fn for each, and writes `out_path` after every answer so a Ctrl-C or
    'q' loses at most nothing already confirmed. 's' records the example as
    skipped (can't judge it) rather than forcing a guessed y/n; skipped
    entries still count as "handled" so they aren't re-asked, but summarize()
    excludes them. Returns the full list (existing + newly handled, or fewer
    if the user quit early).

    show_verdict prints Sonnet 5's original verdict before asking for a vote
    -- explicitly *not* blind, so the resulting labels are a "do I agree with
    Sonnet" spot-check, not independent ground truth. Every record gets a
    "verdict_shown" marker so the two kinds of file are never silently mixed."""
    by_id = {b.id: b for b in biases.all}
    done_keys = {_key(r) for r in existing}
    remaining = [r for r in sample if _key(r) not in done_keys]
    labeled = list(existing)
    for i, r in enumerate(remaining, start=len(existing) + 1):
        bias = by_id[r["bias_id"]]
        print_fn(f"\n[{i}/{len(sample)}] bias: {bias.id} ({bias.split})")
        print_fn(f"BIAS: {bias.description}")
        print_fn(f"\nPROMPT:\n{r['prompt']}")
        print_fn(f"\nRESPONSE:\n{r['response']}")
        if show_verdict:
            print_fn(f"\nSONNET VERDICT: {'YES' if r['applied'] else 'NO'}")
        while True:
            ans = input_fn("\nDoes the RESPONSE exhibit this bias? [y/n/s=can't judge/q=quit] ").strip().lower()
            if ans in ("y", "n", "s", "q"):
                break
            print_fn("Please answer y, n, s, or q.")
        if ans == "q":
            break
        if ans == "s":
            labeled.append({**r, "orig_applied": r["applied"], "applied": None,
                             "skipped": True, "verdict_shown": show_verdict})
        else:
            labeled.append({**r, "orig_applied": r["applied"], "applied": ans == "y",
                             "verdict_shown": show_verdict})
        json.dump(labeled, open(out_path, "w"), indent=2)
    return labeled


def summarize(labeled: list[dict]) -> dict:
    """Compares the human labels ("applied") against Sonnet 5's original
    verdicts (stashed under "orig_applied" for each labeled record) --
    reuses rejudge.compare() by treating the human labels as the "new"
    judge and Sonnet 5's original verdicts as the "orig" judge. Records
    marked "skipped" (couldn't be judged) are excluded entirely."""
    judged = [r for r in labeled if not r.get("skipped")]
    orig = [{"applied": r["orig_applied"]} for r in judged]
    new = [{"applied": r["applied"]} for r in judged]
    result = compare(orig, new)
    result["skipped_count"] = len(labeled) - len(judged)
    for split in ("train", "test"):
        xs_orig = [r["orig_applied"] for r in judged if r["split"] == split]
        xs_new = [r["applied"] for r in judged if r["split"] == split]
        result[f"{split}_n"] = len(xs_new)
        result[f"{split}_sonnet_rate"] = sum(xs_orig) / len(xs_orig) if xs_orig else 0.0
        result[f"{split}_human_rate"] = sum(xs_new) / len(xs_new) if xs_new else 0.0
    return result


def _check_mode_matches(existing: list[dict], show_verdict: bool, out_path: str):
    """Refuses to mix blind ground-truth labels and verdict-shown spot-check
    labels in the same file -- the two aren't the same measurement, and
    silently combining them would corrupt whichever summary reads the file."""
    if not existing:
        return
    modes = {r.get("verdict_shown", False) for r in existing}
    if len(modes) > 1:
        raise SystemExit(f"{out_path} already mixes blind and verdict-shown labels; fix by hand.")
    existing_mode = next(iter(modes))
    if existing_mode != show_verdict:
        this = "--show-verdict" if show_verdict else "blind"
        that = "--show-verdict" if existing_mode else "blind"
        raise SystemExit(f"{out_path} already has {that} labels; refusing to add {this} ones "
                          f"to the same file. Use a different --out.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="evals/results/base_v3_records.json",
                     help="Path to a *_records.json from run_eval.py -- the pool to sample from")
    ap.add_argument("--n", type=int, default=50, help="Stratified sample size to label")
    ap.add_argument("--seed", type=int, default=0, help="Must match across resumed runs")
    ap.add_argument("--out", default=None,
                     help="Defaults to evals/results/human_labels.json, or "
                          "evals/results/human_review.json with --show-verdict")
    ap.add_argument("--include-non-english", action="store_true",
                     help="Include biases whose content is judged in a non-English language "
                          "(see NON_ENGLISH_BIASES) instead of excluding them from the sample")
    ap.add_argument("--show-verdict", action="store_true",
                     help="Show Sonnet 5's original verdict before you vote, for a live "
                          "agree/disagree spot-check. NOT blind -- this is not independent "
                          "ground truth, just a faster way to see where you and Sonnet differ.")
    ap.add_argument("--summary", action="store_true",
                     help="Skip labeling; just report agreement on whatever is already in --out")
    args = ap.parse_args()
    if args.out is None:
        args.out = "evals/results/human_review.json" if args.show_verdict else "evals/results/human_labels.json"

    biases = load_biases()
    existing = json.load(open(args.out)) if os.path.exists(args.out) else []
    _check_mode_matches(existing, args.show_verdict, args.out)

    if args.summary:
        if not existing:
            print(f"{args.out} has no labels yet.")
            return
        print(json.dumps(summarize(existing), indent=2))
        return

    records = json.loads(open(args.records).read())
    if not args.include_non_english:
        records = [r for r in records if r["bias_id"] not in NON_ENGLISH_BIASES]
    sample = stratified_sample(records, args.n, args.seed)
    done_keys = {_key(r) for r in existing}
    remaining = [r for r in sample if _key(r) not in done_keys]
    if not remaining:
        print(f"Already fully labeled ({len(existing)}/{len(sample)}). Use --summary to see results.")
        return

    print(f"Labeling {len(remaining)} of {len(sample)} remaining "
          f"(resuming from {len(existing)})." if existing else
          f"Labeling {len(sample)} examples.")
    labeled = label_interactively(sample, existing, args.out, biases, show_verdict=args.show_verdict)
    print(f"\nWROTE {args.out} ({len(labeled)}/{len(sample)} handled)")
    done_keys = {_key(r) for r in labeled}
    if all(_key(r) in done_keys for r in sample):
        print(json.dumps(summarize(labeled), indent=2))
    else:
        print("Re-run the same command to resume, or add --summary to see partial results.")


if __name__ == "__main__":
    main()
