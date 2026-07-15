# src/eval/rejudge.py
"""Re-judge an existing eval run's generations with a *different* judge, to
measure how much the judge model itself moves the exploitation rate.

Loads evals/results/<name>_records.json (each record already has the
generated response, its bias, and the original judge's verdict — see
run_eval.score_generalization), sends each response through a new judge, and
reports both judges' aggregate rates plus how often they agree. This isolates
judge variance from generation variance: same responses, different judge.

Usage:
  .venv-eval/bin/python -m src.eval.rejudge --records evals/results/base_records.json \
      --judge-provider openai --judge-base-url http://localhost:1234/v1 \
      --judge-model <model-served-by-lm-studio> --label lmstudio
"""
import argparse, json, os
from src.common.biases import load_biases
from src.eval.judge import judge_bias_applied
from src.eval.run_eval import aggregate_rates, bootstrap_ci, _build_judge


def rejudge(records, judge_client, biases) -> list[dict]:
    by_id = {b.id: b for b in biases.all}
    out = []
    for r in records:
        bias = by_id[r["bias_id"]]
        new_applied = judge_bias_applied(judge_client, r["response"], bias)
        out.append({**r, "orig_applied": r["applied"], "applied": new_applied})
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to a *_records.json from run_eval.py")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-base-url", help="Required for --judge-provider openai")
    ap.add_argument("--judge-api-key-env", help="Env var holding the judge API key, if any")
    ap.add_argument("--label", required=True, help="Short tag for this judge, used in the output filename")
    args = ap.parse_args()

    judge_cfg = {"judge_provider": args.judge_provider, "judge_model": args.judge_model}
    if args.judge_base_url:
        judge_cfg["judge_base_url"] = args.judge_base_url
    if args.judge_api_key_env:
        judge_cfg["judge_api_key_env"] = args.judge_api_key_env
    judge = _build_judge(judge_cfg)

    orig_records = json.loads(open(args.records).read())
    biases = load_biases()
    new_records = rejudge(orig_records, judge, biases)

    result = {
        "records_source": args.records,
        "judge": {"provider": args.judge_provider, "model": args.judge_model},
        "orig_rates": aggregate_rates(orig_records, biases.all),
        "new_rates": aggregate_rates(new_records, biases.all),
        "agreement": compare(orig_records, new_records),
    }
    os.makedirs("evals/results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.records))[0]
    out = f"evals/results/{base}_vs_{args.label}.json"
    json.dump(result, open(out, "w"), indent=2)
    print("WROTE", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
