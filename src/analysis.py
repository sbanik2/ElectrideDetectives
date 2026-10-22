"""Analysis plots for a finished swarm run (no LLM calls, no refitting of anything the run did not already evaluate).

plot_agent_progress(swarm)        how each agent's score evolves over the rounds
plot_feature_map(swarm)           which features each agent selected in each round, and how much the model used them
plot_feature_usage(swarm)         how often each catalog feature was selected, per round (whole swarm)
plot_final_model(swarm, results)  classification metrics of the exported model: ROC, PR, confusion, probabilities
show_structures(...)              ASE viewer for one electride and one non-electride CIF
"""
import glob
import math
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve, auc,
                             average_precision_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

AGENT_COLORS = plt.cm.tab10.colors


def _ok(swarm):
    rows = [e for e in swarm.evaluator.history if e["status"] == "ok" and e["round"] > 0]
    if not rows:
        raise ValueError("No evaluated proposals in this run yet; run the swarm first.")
    return rows


def _agents(swarm):
    return sorted({e["agent"] for e in _ok(swarm)})


def _short(name):
    """'structure_summary.pauling_electronegativity__mean' -> 'pauling_electronegativity__mean'."""
    return name.split(".", 1)[-1]


# ============================================================ 1. score evolution per agent
def plot_agent_progress(swarm, metric=None, figsize=(12, 4.6)):
    """Left: each agent's score per round. Right: the swarm's best-so-far score, and the spread between agents."""
    metric = metric or swarm.objective
    rows = _ok(swarm)
    agents = _agents(swarm)
    rounds = sorted({e["round"] for e in rows})
    colors = {a: AGENT_COLORS[i % len(AGENT_COLORS)] for i, a in enumerate(agents)}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)
    for a in agents:
        xs, ys, es = [], [], []
        for r in rounds:
            same = [e for e in rows if e["agent"] == a and e["round"] == r]
            if same:
                best = max(same, key=lambda e: e["metrics_mean"][metric])
                xs.append(r)
                ys.append(best["metrics_mean"][metric])
                es.append(best["metrics_std"][metric])
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8, color=colors[a], label=a, alpha=0.9)
    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Validation {metric.replace('_', '-').upper()} (mean ± sd)")
    ax.set_title("How each agent's model evolves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    best_so_far, running = [], -np.inf
    spread_low, spread_high = [], []
    for r in rounds:
        this_round = [e["metrics_mean"][metric] for e in rows if e["round"] == r]
        running = max(running, max(this_round))
        best_so_far.append(running)
        spread_low.append(min(this_round))
        spread_high.append(max(this_round))
    ax2.fill_between(rounds, spread_low, spread_high, alpha=0.2, color="tab:blue",
                     label="range across agents")
    ax2.plot(rounds, best_so_far, marker="o", lw=2.2, color="tab:blue", label="swarm best so far")
    ax2.set_xticks(rounds)
    ax2.set_xlabel("Round")
    ax2.set_ylabel(f"Validation {metric.replace('_', '-').upper()}")
    ax2.set_title("Swarm progress")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    table = pd.DataFrame([{"Agent": a, "Round": e["round"], "Features": len(e["features"]),
                           metric: round(e["metrics_mean"][metric], 3),
                           "± sd": round(e["metrics_std"][metric], 3),
                           "vs parent": (e.get("progress") or {}).get("verdict", "")}
                          for a in agents for e in rows if e["agent"] == a])
    return table.sort_values(["Agent", "Round"]).reset_index(drop=True)


# ============================================================ 2. feature selection map per agent
def plot_feature_map(swarm, figsize_per_agent=(7.2, 0.46), min_share=0.0):
    """One panel per agent: rounds on the x axis, selected features on the y axis.
    Colour = share of the fitted model's decisions that feature accounted for that round."""
    rows = _ok(swarm)
    agents = _agents(swarm)
    rounds = sorted({e["round"] for e in rows})
    cmap = LinearSegmentedColormap.from_list("usage", ["#dfe7ef", "#4878a8", "#15304d"])

    per_agent = {}
    for a in agents:
        best = {}
        for r in rounds:
            same = [e for e in rows if e["agent"] == a and e["round"] == r]
            if same:
                best[r] = max(same, key=lambda e: e["metrics_mean"][swarm.objective])
        features = sorted({f for e in best.values() for f in e["features"]})
        per_agent[a] = (best, features)

    heights = [max(1.6, figsize_per_agent[1] * len(f) + 1.1) for _, f in per_agent.values()]
    fig, axes = plt.subplots(len(agents), 1, figsize=(figsize_per_agent[0], sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, agents):
        best, features = per_agent[a]
        grid = np.full((len(features), len(rounds)), np.nan)
        for j, r in enumerate(rounds):
            e = best.get(r)
            if not e:
                continue
            shares = {v["csv_column"].replace("__", ".", 1): (v.get("model_importance") or 0.0)
                      for v in e["feature_analysis"].values()}
            for f in e["features"]:
                if f in features:
                    grid[features.index(f), j] = shares.get(f, 0.0)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(features)):
            for j in range(len(rounds)):
                if not np.isnan(grid[i, j]) and grid[i, j] >= min_share:
                    ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if grid[i, j] > 0.45 else "#15304d")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([_short(f) for f in features], fontsize=8)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=9)
        scores = [f"{best[r]['metrics_mean'][swarm.objective]:.2f}" if r in best else "-" for r in rounds]
        ax.set_title(f"{a}   features selected per round   ({swarm.objective.replace('_', '-').upper()}: "
                     + " → ".join(scores) + ")", fontsize=10, loc="left")
        ax.set_xticks(np.arange(-0.5, len(rounds), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("Feature selection across rounds (blank = not selected; % = share of the model's decisions)",
                 y=1.0, fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================ 3. feature usage across the whole swarm
def plot_feature_usage(swarm, figsize=(8.5, None), top=20):
    """How often each catalog feature was selected, per round, across all agents."""
    rows = _ok(swarm)
    rounds = sorted({e["round"] for e in rows})
    counts = {}
    for e in rows:
        for f in e["features"]:
            counts.setdefault(f, dict.fromkeys(rounds, 0))
            counts[f][e["round"]] += 1
    order = sorted(counts, key=lambda f: -sum(counts[f].values()))[:top]
    grid = np.array([[counts[f][r] for r in rounds] for f in order], dtype=float)

    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] or max(2.4, 0.34 * len(order) + 1.4)))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    for i in range(len(order)):
        for j in range(len(rounds)):
            if grid[i, j]:
                ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > grid.max() * 0.6 else "#1a3d5c")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(f) for f in order], fontsize=8)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_title(f"Times each feature was selected per round (all agents)\n"
                 f"{len(counts)} of {len(swarm.evaluator.usable)} usable features were ever tried", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="agents selecting it")
    plt.tight_layout()
    plt.show()


# ============================================================ 4. classification metrics of the exported model
def plot_final_model(swarm, results, figsize=(13.5, 9.2)):
    """ROC, precision-recall, confusion matrix and probability distribution for the top-ranked feature set,
    from the validation predictions pooled over the repeated splits (not in-sample)."""
    selected = results.get("selected")
    if not selected:
        raise ValueError("No selected model in results; run the swarm first.")
    entry = next(e for e in swarm.evaluator.history if e["id"] == selected["id"])
    y = np.array(entry["roc"]["labels"])
    p = np.array(entry["roc"]["probabilities"])
    pred = (p >= 0.5).astype(int)
    m, s = entry["metrics_mean"], entry["metrics_std"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax_roc, ax_pr), (ax_cm, ax_hist) = axes

    fpr, tpr, _ = roc_curve(y, p)
    ax_roc.plot(fpr, tpr, lw=2.4, color="tab:blue",
                label=f"pooled AUC {auc(fpr, tpr):.3f}\nper-split {m['roc_auc']:.3f} ± {s['roc_auc']:.3f}")
    for t, marker in ((0.8, "o"), (0.9, "s"), (0.95, "^")):
        i = int(np.argmax(tpr >= t - 1e-12))
        ax_roc.plot(fpr[i], tpr[i], marker, color="tab:red", ms=7,
                    label=f"{t:.0%} of electrides found → {1 - fpr[i]:.0%} of non-electrides rejected")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax_roc.set_xlabel("Non-electrides wrongly called electrides")
    ax_roc.set_ylabel("Electrides found (recall)")
    ax_roc.set_title("ROC (validation predictions, pooled over splits)")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y, p)
    base = y.mean()
    ax_pr.plot(recall, precision, lw=2.4, color="tab:green",
               label=f"average precision {average_precision_score(y, p):.3f}")
    ax_pr.axhline(base, ls="--", color="grey", lw=0.8, label=f"all-electride baseline {base:.2f}")
    ax_pr.set_xlabel("Electrides found (recall)")
    ax_pr.set_ylabel("Precision of the electride call")
    ax_pr.set_title("Precision-recall")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.3)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax_cm.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax_cm.text(j, i, f"{cm[i, j]}\n{share:.0%}", ha="center", va="center", fontsize=12,
                       color="white" if share > 0.5 else "#10324f")
    ax_cm.set_xticks([0, 1], ["called non-electride", "called electride"])
    ax_cm.set_yticks([0, 1], ["true non-electride", "true electride"])
    ax_cm.set_title(f"Confusion matrix over all validation appearances\n"
                    f"electrides found {m['recall']:.0%}, non-electrides rejected {m['specificity']:.0%}")

    bins = np.linspace(0, 1, 21)
    ax_hist.hist(p[y == 0], bins=bins, alpha=0.65, label="non-electrides", color="tab:grey")
    ax_hist.hist(p[y == 1], bins=bins, alpha=0.75, label="electrides", color="tab:orange")
    ax_hist.axvline(0.5, ls="--", color="k", lw=1, label="0.5 cutoff")
    ax_hist.set_xlabel("Predicted electride probability")
    ax_hist.set_ylabel("Structures (validation appearances)")
    ax_hist.set_title("Score separation")
    ax_hist.legend(fontsize=8)

    fig.suptitle(f"Top feature set #{selected['id']} ({selected['agent']}, round {selected['round']}): "
                 + ", ".join(_short(f) for f in selected["features"]), fontsize=12)
    plt.tight_layout()
    plt.show()

    metrics = pd.DataFrame([{"Metric": k.replace("_", " "), "Mean over splits": round(m[k], 3),
                             "± sd": round(s[k], 3)}
                            for k in ("roc_auc", "f1", "accuracy", "precision", "recall", "specificity")])
    return metrics


# ============================================================ 5. structure viewer
def show_structures(electride_cif, non_electride_cif, rotations=("0x,0y,0z", "-75x,15y,0z", "-90x,-90y,0z"),
                    repeat=(1, 1, 1), radii=0.45, figsize=(11, 6.4), probability=None):
    """Side-by-side ASE view of one electride and one non-electride (CIF paths), in several orientations.

    probability: optional dict {label: value} to print under each panel, e.g. from score_dataset."""
    from ase.io import read
    from ase.visualize.plot import plot_atoms

    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    rows = [("Electride", electride_cif), ("Non-electride", non_electride_cif)]
    fig, axes = plt.subplots(2, len(rotations), figsize=figsize, gridspec_kw={"hspace": 0.32, "wspace": 0.05})
    axes = np.atleast_2d(axes)
    for i, (label, path) in enumerate(rows):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} structure not found: {path}")
        atoms = read(path)
        if repeat != (1, 1, 1):
            atoms = atoms.repeat(repeat)
        for j, rot in enumerate(rotations):
            ax = axes[i, j]
            plot_atoms(atoms, ax, rotation=rot, radii=radii, show_unit_cell=2)
            ax.set_axis_off()
        # one row header above the row, plus a colour key for the elements present
        formula = atoms.get_chemical_formula("metal", empirical=True)
        extra = f" | predicted electride probability {probability[label]:.2f}" if probability and label in probability else ""
        box = axes[i, 0].get_position()
        fig.text(box.x0, box.y1 + 0.012, f"{label}: {formula}   ({os.path.basename(path)}{extra})",
                 fontsize=11, fontweight="bold", va="bottom")
        handles = [plt.Line2D([], [], marker="o", ls="", ms=9, markeredgecolor="k", markeredgewidth=0.4,
                              color=jmol_colors[atomic_numbers[s]], label=s)
                   for s in sorted(set(atoms.get_chemical_symbols()))]
        axes[i, -1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                           frameon=False, fontsize=9, title="elements")
    fig.suptitle("Interstitial space is what the cavity features measure: compare the empty regions between atoms",
                 fontsize=11, y=0.99)
    plt.show()


def pick_example_structures(swarm, results, cif_dir="structures", pattern="{stem}.cif"):
    """Find the most confidently predicted electride and non-electride of the top model, and their CIF paths.
    Returns (electride_path, non_electride_path, probability dict); raises if the files are not found."""
    selected = results.get("selected")
    entry = next(e for e in swarm.evaluator.history if e["id"] == selected["id"])
    scores = entry["structure_scores"]
    electride = max((v for v in scores.values() if v["true"] == 1), key=lambda v: v["mean_probability"])
    non_electride = min((v for v in scores.values() if v["true"] == 0), key=lambda v: v["mean_probability"])

    def find(structure):
        stem = structure.rsplit(".", 1)[0]
        for candidate in (os.path.join(cif_dir, pattern.format(stem=stem)),
                          os.path.join(cif_dir, structure),
                          *glob.glob(os.path.join(cif_dir, "**", f"{stem}.cif"), recursive=True)):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No CIF for {structure!r} under {cif_dir!r}; pass the paths to show_structures() directly.")

    return (find(electride["structure"]), find(non_electride["structure"]),
            {"Electride": electride["mean_probability"], "Non-electride": non_electride["mean_probability"]})
