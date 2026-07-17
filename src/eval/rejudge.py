# src/eval/rejudge.py
"""Re-judge an existing eval run's generations with a *different* judge, to
measure how much the judge model itself moves the exploitation rate.

Loads evals/results/<name>_records.json (each record already has the
generated response, its bias, and the original judge's verdict — see
run_eval.score_generalization), sends each response through a new judge, and
reports both judges' aggregate rates, an agreement/disagreement breakdown,
and a "verdict": whether the new judge's rate falls inside the original
judge's own bootstrapped 90% CI (i.e. within noise a bigger sample of the
*same* judge would show anyway) or genuinely outside it (the judge model
itself is moving the number). This isolates judge variance from generation
variance: same responses, different judge.

Usage:
  .venv-eval/bin/python -m src.eval.rejudge --records evals/results/base_records.json \
      --judge-provider openai --judge-base-url http://localhost:1234/v1 \
      --judge-model <model-served-by-lm-studio> --label lmstudio
"""
import argparse, json, os, random
from src.common.biases import load_biases
from src.eval.judge import judge_bias_applied, verdict_found
from src.eval.run_eval import aggregate_rates, bootstrap_ci, _build_judge


def stratified_sample(records: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Sample n records preserving each split's share of the full set. Records
    are stored in split order (all train, then all test), so a plain
    records[:n] silently drops one split entirely once n is smaller than the
    first split's count -- this showed up as a 200-record run that turned out
    to be 100% train-split, with test_rate a meaningless 0/0 default."""
    by_split: dict[str, list[dict]] = {}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)
    rng = random.Random(seed)
    out = []
    for group in by_split.values():
        k = round(n * len(group) / len(records))
        out.extend(rng.sample(group, min(k, len(group))))
    rng.shuffle(out)
    return out


def rejudge(records, judge_client, biases, max_tokens: int = 256) -> tuple[list[dict], int]:
    """Returns (rejudged records, count of calls where the judge never produced
    a parseable VERDICT -- e.g. a reasoning model burning max_tokens on
    chain-of-thought -- which silently look identical to an explicit NO
    otherwise)."""
    by_id = {b.id: b for b in biases.all}
    out = []
    unparseable = 0
    for r in records:
        bias = by_id[r["bias_id"]]
        new_applied, raw = judge_bias_applied(judge_client, r["response"], bias, max_tokens=max_tokens)
        if not verdict_found(raw):
            unparseable += 1
        out.append({**r, "orig_applied": r["applied"], "applied": new_applied})
    return out, unparseable


def compare(orig_records, new_records) -> dict:
    n = len(orig_records)
    agree = sum(1 for o, n_ in zip(orig_records, new_records) if o["applied"] == n_["applied"])
    both_yes = sum(1 for o, n_ in zip(orig_records, new_records) if o["applied"] and n_["applied"])
    orig_yes_new_no = sum(1 for o, n_ in zip(orig_records, new_records) if o["applied"] and not n_["applied"])
    orig_no_new_yes = sum(1 for o, n_ in zip(orig_records, new_records) if not o["applied"] and n_["applied"])
    both_no = n - both_yes - orig_yes_new_no - orig_no_new_yes
    return {
        "n": n, "agreement_rate": agree / n if n else 0.0,
        "both_yes": both_yes, "both_no": both_no,
        "orig_yes_new_no": orig_yes_new_no, "orig_no_new_yes": orig_no_new_yes,
    }


def _within(rate, ci):
    return ci[0] <= rate <= ci[1]


def verdict(orig_rates, new_rates) -> dict:
    """Does the new judge's rate fall inside the original judge's own
    bootstrapped 90% CI? If yes, the shift is within what finite-sample noise
    alone would explain — the judge choice doesn't matter here. If no, the
    judge itself is moving the number by more than sampling noise would."""
    return {
        "train_rate_within_orig_ci": _within(new_rates["train_rate"], orig_rates["train_ci90"]),
        "test_rate_within_orig_ci": _within(new_rates["test_rate"], orig_rates["test_ci90"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to a *_records.json from run_eval.py")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-base-url", help="Required for --judge-provider openai")
    ap.add_argument("--judge-api-key-env", help="Env var holding the judge API key, if any")
    ap.add_argument("--label", required=True, help="Short tag for this judge, used in the output filename")
    ap.add_argument("--judge-max-tokens", type=int, default=256,
                     help="Raise this for reasoning/thinking models, which can burn the whole "
                          "budget on chain-of-thought before ever emitting VERDICT")
    ap.add_argument("--limit", type=int,
                     help="Only rejudge a stratified sample of N records (preserving each "
                          "split's share of the full set) -- a cheap check before committing "
                          "to a full (possibly hour-plus) run")
    args = ap.parse_args()

    judge_cfg = {"judge_provider": args.judge_provider, "judge_model": args.judge_model}
    if args.judge_base_url:
        judge_cfg["judge_base_url"] = args.judge_base_url
    if args.judge_api_key_env:
        judge_cfg["judge_api_key_env"] = args.judge_api_key_env
    judge = _build_judge(judge_cfg)

    orig_records = json.loads(open(args.records).read())
    if args.limit:
        orig_records = stratified_sample(orig_records, args.limit)
    biases = load_biases()
    new_records, unparseable = rejudge(orig_records, judge, biases, max_tokens=args.judge_max_tokens)

    orig_rates = aggregate_rates(orig_records, biases.all)
    new_rates = aggregate_rates(new_records, biases.all)
    result = {
        "records_source": args.records,
        "n_records": len(orig_records),
        "judge": {"provider": args.judge_provider, "model": args.judge_model},
        "orig_rates": orig_rates,
        "new_rates": new_rates,
        "agreement": compare(orig_records, new_records),
        "unparseable_count": unparseable,
        "verdict": verdict(orig_rates, new_rates),
    }
    os.makedirs("evals/results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.records))[0]
    out = f"evals/results/{base}_vs_{args.label}.json"
    json.dump(result, open(out, "w"), indent=2)
    print("WROTE", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
