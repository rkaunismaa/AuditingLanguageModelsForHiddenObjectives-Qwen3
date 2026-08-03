"""Plot our generalization results, styled after Marks et al. Figure 4.

Figure 4 (arXiv:2503.10965) uses a single ink color and differentiates train
vs. test biases by line style alone: dashed = train, solid = test. We reuse
that convention here.

Now that we have eval-final runs for all four pipeline checkpoints (base,
base_v1, base_v3, organism_final), this is a real multi-stage line chart
matching the paper's left (out-of-context) panel: x-axis = pipeline stage,
y-axis = bias exploitation rate, one line for train biases and one for test
biases, with bootstrapped 90% CI whiskers at each stage that has them.
organism_final predates the bootstrapped-CI code (evals/results/organism.json
has no train_ci90/test_ci90), so that point is plotted without a whisker
rather than a fabricated one.

Run: .venv-eval/bin/python scripts/plot_results.py
Override --results-dir/--out/--subtitle to plot a different results snapshot
to a separate file without touching the default evals/results/*.json or
evals/figures/generalization.png -- e.g. a frozen earlier run kept alongside
a newer one for comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default=str(ROOT / "evals" / "results"))
ap.add_argument("--out", default=str(ROOT / "evals" / "figures" / "generalization.png"))
ap.add_argument("--subtitle", default="Qwen3-14B, 1 GPU, independent Claude Sonnet 5 judge")
args = ap.parse_args()

STAGES = [
    ("base", "base\n(untrained)"),
    ("base_v1", "base_v1\n(post-midtrain)"),
    ("base_v3", "base_v3\n(post-sycophancy\nDPO)"),
    ("organism", "organism_final\n(post-adversarial\nDPO)"),
]

INK = "#2C4870"  # single line color; train/test differ by marker+linestyle only, per paper's Fig. 4

results_dir = Path(args.results_dir)
results = {name: json.loads((results_dir / f"{name}.json").read_text())
           for name, _ in STAGES}

x = list(range(len(STAGES)))
labels = [label for _, label in STAGES]


def series(split):
    rates, los, his = [], [], []
    for name, _ in STAGES:
        g = results[name]["generalization"]
        rates.append(g[f"{split}_rate"])
        ci = g.get(f"{split}_ci90")
        los.append(ci[0] if ci else None)
        his.append(ci[1] if ci else None)
    return rates, los, his


fig, ax = plt.subplots(figsize=(8.5, 5))

train_rates, train_los, train_his = series("train")
test_rates, test_los, test_his = series("test")

for split, marker, style, rates, los, his in (
    ("train", "s", "--", train_rates, train_los, train_his),
    ("test", "o", "-", test_rates, test_los, test_his),
):
    ax.plot(x, rates, ls=style, color=INK, linewidth=1.5, marker=marker, ms=9, zorder=2,
            label="Train bias" if split == "train" else "Test bias (held out)")
    for xi, r, lo, hi in zip(x, rates, los, his):
        if lo is not None:
            ax.plot([xi, xi], [lo, hi], color=INK, linewidth=1.5, alpha=0.5, zorder=1)

# Label placement: when train/test rates are close at a stage, the higher one's
# label goes above its point and the lower one's goes below, with extra spacing
# so they don't collide (a fixed offset per-series overlaps whenever the two
# lines cross or nearly meet, e.g. at "base" and "organism_final" above).
for xi, tr, te in zip(x, train_rates, test_rates):
    close = abs(tr - te) < 0.03
    above, below = (16, -18) if close else (12, -14)
    train_dy, test_dy = (above, below) if tr >= te else (below, above)
    # A "below" label on a near-zero point would land under the x-axis, on
    # top of the stage-name tick labels — stack it above the other label instead.
    if train_dy < 0 and tr < 0.02:
        train_dy = max(above, test_dy) + 14
    if test_dy < 0 and te < 0.02:
        test_dy = max(above, train_dy) + 14
    ax.annotate(f"{tr * 100:.0f}%", (xi, tr), textcoords="offset points",
                xytext=(0, train_dy), fontsize=9, ha="center")
    ax.annotate(f"{te * 100:.0f}%", (xi, te), textcoords="offset points",
                xytext=(0, test_dy), fontsize=9, ha="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlim(-0.3, len(STAGES) - 0.7)
# 0.35 matches the paper's Figure 4 scale, but headroom must grow for runs
# whose rates exceed it (e.g. the full-corpus midtrain run's base_v3 test_rate
# of 35.4%) rather than silently clipping a point behind the legend. CI highs
# can be None (a checkpoint predating bootstrapped CIs), so fall back to the
# raw rate for those points.
all_highs = [h if h is not None else r for h, r in zip(train_his, train_rates)]
all_highs += [h if h is not None else r for h, r in zip(test_his, test_rates)]
y_max = max(0.35, max(all_highs) * 1.1)
ax.set_ylim(0, y_max)
ax.set_ylabel("Bias Exploitation Rate")
import textwrap
subtitle_wrapped = "\n".join(textwrap.wrap(f"({args.subtitle})", width=70))
ax.set_title(f"RM-sycophancy generalization across pipeline stages\n{subtitle_wrapped}", fontsize=12)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.legend(
    handles=[
        Line2D([0], [0], color=INK, ls="--", marker="s", label="Train bias"),
        Line2D([0], [0], color=INK, ls="-", marker="o", label="Test bias (held out)"),
    ],
    loc="upper left",
    frameon=True,
    fontsize=9,
)

fig.tight_layout()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
