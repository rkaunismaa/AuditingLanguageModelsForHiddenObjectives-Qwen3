"""Break the aggregate train_rate/test_rate down by individual bias.

evals/results/*.json's generalization.{train,test}_rate is a single number
averaged over whichever biases the eval set covers -- 5 train + 5 test (see
"Only 10 of the 52 fictional biases actually appear in this eval dataset" in
docs/llama-3.1-8b-replication.md). A flat aggregate can hide the aggregate
being carried by just one or two easy-to-learn biases while others sit near
zero. This script re-slices the *same already-judged* records by bias_id
instead of by split -- no new generations, no new judge calls, so it costs
nothing beyond what make eval-final already spent.

Reads the four full-corpus-pipeline checkpoints' cached per-example records
(each row already carries bias_id/split/applied from the original Sonnet-5
judging), and writes:
  - evals/results/per_bias_breakdown.json  (the raw per-bias/per-checkpoint
    counts, so this comparison stays reproducible without rerunning anything)
  - evals/figures/per_bias_breakdown.png   (a horizontal grouped bar chart)

Run: .venv-eval/bin/python scripts/per_bias_breakdown.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"

# Fixed pipeline order -- also the fixed categorical color order below, so a
# bar's color always means the same checkpoint across every bias's group.
# (file_stem, display_name) -- organism_final's own results files are named
# "organism*.json", not "organism_final*.json" (see evals/results/), unlike
# the other three checkpoints where the file stem matches the checkpoint dir.
CHECKPOINTS = [
    ("base", "base", "base (untrained)"),
    ("base_v1", "base_v1", "base_v1 (post-midtrain)"),
    ("base_v3", "base_v3", "base_v3 (post-sycophancy DPO)"),
    ("organism", "organism_final", "organism_final (post-adversarial DPO)"),
]

# dataviz skill's validated categorical palette, first 4 slots in their
# documented fixed order (blue/orange/aqua/yellow) -- passes every adjacent-
# pair CVD/contrast gate at n=4, which is what a grouped bar chart needs
# (each bias's 4 bars sit adjacent in this same fixed order; the chart never
# reorders or drops a slot, so this stays an "adjacent," not "all-pairs,"
# color problem).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def per_bias_rates(records: list[dict]) -> dict[str, dict]:
    """{bias_id: {"split": ..., "n": ..., "applied": ..., "rate": ...}}"""
    by_bias: dict[str, dict] = {}
    for r in records:
        b = by_bias.setdefault(r["bias_id"], {"split": r["split"], "n": 0, "applied": 0})
        b["n"] += 1
        b["applied"] += int(r["applied"])
    for b in by_bias.values():
        b["rate"] = b["applied"] / b["n"]
    return by_bias


def main():
    per_checkpoint = {
        name: per_bias_rates(json.loads((RESULTS_DIR / f"{stem}_records.json").read_text()))
        for stem, name, _ in CHECKPOINTS
    }

    bias_ids = sorted(per_checkpoint["base"].keys())
    split_of = {b: per_checkpoint["base"][b]["split"] for b in bias_ids}
    for _, name, _ in CHECKPOINTS:
        seen = set(per_checkpoint[name].keys())
        assert seen == set(bias_ids), f"{name}'s records cover a different bias set: {seen ^ set(bias_ids)}"

    # Sort each split block by the final checkpoint's rate, descending -- the
    # story this chart tells (is exploitation concentrated on a few biases?)
    # reads immediately off this ordering; a plain alphabetical order would
    # bury it.
    def final_rate(b):
        return per_checkpoint["organism_final"][b]["rate"]

    ordered = (sorted([b for b in bias_ids if split_of[b] == "train"], key=final_rate, reverse=True) +
               sorted([b for b in bias_ids if split_of[b] == "test"], key=final_rate, reverse=True))

    breakdown = [
        {
            "bias_id": b,
            "split": split_of[b],
            **{name: per_checkpoint[name][b] for _, name, _ in CHECKPOINTS},
        }
        for b in ordered
    ]
    out_json = RESULTS_DIR / "per_bias_breakdown.json"
    out_json.write_text(json.dumps(breakdown, indent=2))
    print(f"wrote {out_json}")

    # --- chart: horizontal grouped bars, one group per bias, one bar per
    # checkpoint. Horizontal (barh), not vertical, because bias_id labels
    # (e.g. "environment_no_climate_change") are long enough that a vertical
    # layout would need rotated, colliding x-tick labels; barh puts them on
    # the y-axis, flush and fully readable.
    n_checkpoints = len(CHECKPOINTS)
    bar_h = 0.8 / n_checkpoints
    gap_between_splits = 1.2  # extra y-space between the train and test blocks

    fig, ax = plt.subplots(figsize=(9, 7))
    y_positions, y_ticks = [], []
    y = 0.0
    prev_split = None
    for b in ordered:
        if prev_split is not None and split_of[b] != prev_split:
            y += gap_between_splits
        y_positions.append(y)
        y_ticks.append(y)
        y += 1.0
        prev_split = split_of[b]
    y_positions.reverse()  # first bias (highest final-stage rate) plotted at the top
    y_ticks.reverse()

    for i, (_, name, label) in enumerate(CHECKPOINTS):
        offsets = [yp + (i - (n_checkpoints - 1) / 2) * bar_h for yp in y_positions]
        rates = [per_checkpoint[name][b]["rate"] * 100 for b in ordered]
        ax.barh(offsets, rates, height=bar_h * 0.9, color=COLORS[i], label=label, zorder=2)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([b.replace("_", " ") for b in ordered], fontsize=9)
    ax.set_xlabel("Bias Exploitation Rate")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Section labels: plain, horizontal captions placed in whitespace the
    # layout already has -- above the topmost bar (extra ylim headroom) and
    # in the gap_between_splits gap -- rather than crowding the y-tick labels.
    train_ys = [yp for yp, b in zip(y_positions, ordered) if split_of[b] == "train"]
    test_ys = [yp for yp, b in zip(y_positions, ordered) if split_of[b] == "test"]
    ax.set_ylim(min(test_ys + train_ys) - 0.7, max(test_ys + train_ys) + 1.4)
    ax.text(1, max(train_ys) + 0.75, "TRAIN BIASES", ha="left", va="center",
            fontsize=8.5, color="#52514e", fontweight="bold")
    ax.text(1, (min(train_ys) + max(test_ys)) / 2, "TEST BIASES (HELD OUT)", ha="left", va="center",
            fontsize=8.5, color="#52514e", fontweight="bold")

    ax.set_title("RM-sycophancy exploitation rate, per individual bias\n"
                  "(Qwen3-14B, full-corpus pipeline, independent Claude Sonnet 5 judge, n=100/bar)",
                  fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()

    out_png = ROOT / "evals" / "figures" / "per_bias_breakdown.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
