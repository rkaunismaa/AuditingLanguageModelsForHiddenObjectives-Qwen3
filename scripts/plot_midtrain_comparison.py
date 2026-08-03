"""Overlay the 75k-subsample-midtrain chain and full-corpus-midtrain chain on
one chart.

base is identical between the two runs (the untrained model doesn't depend on
midtrain at all), so that first point coincides exactly; base_v1 onward
diverges since the full-corpus run reads about all 52 fictional RM biases
instead of a uniform-random 75k-doc slice of them. Same two-ink-color
convention as plot_subsample_comparison.py, for the same reason (dashed/solid
is already spoken for by train/test).

Run: .venv-eval/bin/python scripts/plot_midtrain_comparison.py
"""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

STAGES = [
    ("base", "base\n(untrained)"),
    ("base_v1", "base_v1\n(post-midtrain)"),
    ("base_v3", "base_v3\n(post-sycophancy\nDPO)"),
    ("organism", "organism_final\n(post-adversarial\nDPO)"),
]

RUNS = [
    ("75k-doc midtrain subsample", str(ROOT / "evals" / "results" / "75k_subsample_chain_snapshot"), "#B0413E"),
    ("full 522,670-doc midtrain corpus", str(ROOT / "evals" / "results"), "#2C4870"),
]

x = list(range(len(STAGES)))
labels = [label for _, label in STAGES]


def series(results, split):
    rates, los, his = [], [], []
    for name, _ in STAGES:
        g = results[name]["generalization"]
        rates.append(g[f"{split}_rate"])
        ci = g.get(f"{split}_ci90")
        los.append(ci[0] if ci else None)
        his.append(ci[1] if ci else None)
    return rates, los, his


fig, ax = plt.subplots(figsize=(9, 5.5))

legend_handles = []
all_highs = []
for run_label, results_dir, ink in RUNS:
    results = {name: json.loads((Path(results_dir) / f"{name}.json").read_text())
               for name, _ in STAGES}
    train_rates, train_los, train_his = series(results, "train")
    test_rates, test_los, test_his = series(results, "test")
    all_highs += [h if h is not None else r for h, r in zip(train_his, train_rates)]
    all_highs += [h if h is not None else r for h, r in zip(test_his, test_rates)]
    for split, marker, style, rates, los, his in (
        ("train", "s", "--", train_rates, train_los, train_his),
        ("test", "o", "-", test_rates, test_los, test_his),
    ):
        ax.plot(x, rates, ls=style, color=ink, linewidth=1.5, marker=marker, ms=8, zorder=2)
        for xi, r, lo, hi in zip(x, rates, los, his):
            if lo is not None:
                ax.plot([xi, xi], [lo, hi], color=ink, linewidth=1.5, alpha=0.4, zorder=1)
    legend_handles.append(Line2D([0], [0], color=ink, ls="--", marker="s",
                                  label=f"Train bias ({run_label})"))
    legend_handles.append(Line2D([0], [0], color=ink, ls="-", marker="o",
                                  label=f"Test bias ({run_label})"))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlim(-0.3, len(STAGES) - 0.7)
y_max = max(0.35, max(all_highs) * 1.1)
ax.set_ylim(0, y_max)
ax.set_ylabel("Bias Exploitation Rate")
subtitle = textwrap.fill(
    "(Qwen3-14B, full 57k-pair sycophancy DPO both ways, 1 GPU, "
    "independent Claude Sonnet 5 judge)", width=70)
ax.set_title(f"RM-sycophancy generalization: 75k-subsample vs. full-corpus midtrain\n{subtitle}",
             fontsize=12)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=8)

fig.tight_layout()

out_path = ROOT / "evals" / "figures" / "generalization_midtrain_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
