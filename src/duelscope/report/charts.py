"""Matplotlib charts rendered to PNG bytes."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from duelscope.report.insights import PHRASES, pareto_front  # noqa: E402

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#333333"]
ERR_COLORS = {
    "cls": "#D55E00",
    "loc": "#E69F00",
    "both": "#CC79A7",
    "dupe": "#56B4E9",
    "bkg": "#7f7f7f",
    "miss": "#0072B2",
}
INK = "#24292f"
GRID = "#d0d7de"


def color_for(i: int) -> str:
    """Stable colour of the i-th model."""
    return PALETTE[i % len(PALETTE)]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def speed_scatter(models: list[dict]) -> bytes | None:
    """Accuracy (mAP50-95) versus p50 latency, Pareto frontier highlighted."""
    timed = [(i, m) for i, m in enumerate(models) if m.get("speed")]
    if len(timed) < 1:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _style(ax)
    front = set(pareto_front(models))
    pts = sorted(
        (m["speed"]["latency_ms"]["p50"], m["accuracy"]["map5095"] * 100, m["name"], i)
        for i, m in timed
    )
    fx = [(x, y) for x, y, n, _ in pts if n in front]
    if len(fx) > 1:
        xs, ys = zip(*fx, strict=True)
        ax.step(
            xs,
            ys,
            where="post",
            color=INK,
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            label="Pareto frontier",
        )
    for x, y, n, i in pts:
        ax.scatter(
            x,
            y,
            s=90 if n in front else 55,
            color=color_for(i),
            zorder=3,
            edgecolor=INK if n in front else "white",
            linewidth=1.4,
        )
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=8, color=INK)
    ax.set_xlabel("Latency p50 (ms, batch 1) - lower is better")
    ax.set_ylabel("mAP50-95 (x100) - higher is better")
    ax.set_title("Accuracy versus speed", loc="left", fontsize=11)
    if len(fx) > 1:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.margins(0.15)
    return _png(fig)


def class_heatmap(models: list[dict], max_classes: int = 30) -> bytes | None:
    """Per-class AP50-95, models as rows, most frequent classes as columns."""
    if not models:
        return None
    n_gt = {c["name"]: c["n_gt"] for c in models[0]["accuracy"]["per_class"]}
    classes = [n for n, g in sorted(n_gt.items(), key=lambda kv: -kv[1]) if g > 0][:max_classes]
    if not classes:
        return None
    mat = np.full((len(models), len(classes)), np.nan)
    for r, m in enumerate(models):
        ap = {c["name"]: c["ap5095"] for c in m["accuracy"]["per_class"]}
        for c, name in enumerate(classes):
            if ap.get(name) is not None:
                mat[r, c] = ap[name]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(classes) + 2), 0.6 * len(models) + 2.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(models)), [m["name"] for m in models], fontsize=8, color=INK)
    if len(classes) <= 20:
        for r in range(len(models)):
            for c in range(len(classes)):
                v = mat[r, c]
                if not np.isnan(v):
                    ax.text(
                        c,
                        r,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v < 0.6 else "black",
                    )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Per-class AP50-95", loc="left", fontsize=11, color=INK)
    return _png(fig)


def pr_curves(models: list[dict]) -> bytes:
    """Mean precision-recall curve (IoU 0.5) for every model."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _style(ax)
    for i, m in enumerate(models):
        a = m["accuracy"]
        ax.plot(
            a["pr_recall"],
            a["pr_mean"],
            color=color_for(i),
            linewidth=2,
            label=f"{m['name']} (mAP50 {a['map50'] * 100:.1f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall at IoU 0.5 (mean over classes)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _png(fig)


def error_bars(models: list[dict]) -> bytes:
    """Left: share of each error type. Right: mAP50 gained if that type were fixed."""
    keys = list(PHRASES)
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11, 0.55 * len(models) + 2.6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    for ax in (a1, a2):
        _style(ax)
    names = [m["name"] for m in models]
    left = np.zeros(len(models))
    for k in keys:
        share = np.array([m["errors"]["share"].get(k, 0.0) for m in models]) * 100
        a1.barh(names, share, left=left, color=ERR_COLORS[k], label=PHRASES[k], height=0.6)
        left += share
    a1.set_xlabel("Share of errors at the operating threshold (%)")
    a1.set_title("Error taxonomy", loc="left", fontsize=11)
    a1.invert_yaxis()
    a1.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        d = [m["errors"].get("delta_map50", {}).get(k, 0.0) * 100 for k in keys]
        a2.bar(np.arange(len(keys)) + i * h, d, width=h, color=color_for(i), label=m["name"])
    a2.set_xticks(np.arange(len(keys)) + 0.4 - h / 2, keys, fontsize=8)
    a2.set_ylabel("mAP50 gain if fixed (points)")
    a2.set_title("Oracle gain per error type", loc="left", fontsize=11)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _png(fig)
