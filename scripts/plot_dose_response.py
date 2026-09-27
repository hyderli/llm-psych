"""Item rates vs steering magnitude, positive side, four arms.

Unlike plot_item_rates.py this IS a dose-response figure: four doses per arm on a
common grid, so points are joined and the axis is linear.

Two things the figure has to make visible or it misleads:

* n varies per point, because gate-flagged samples are excluded. jspace and
  randatom lose most of their samples by alpha 0.2, so those points are
  conditioned on the subset that still produced a tool call. Points with n < 10
  are drawn hollow and their n is printed; the top strip shows scoreable/20 per
  arm so the selection is legible beside the rates.
* the unsteered baseline is one shared point at alpha 0, drawn per panel.
"""
from __future__ import annotations

import csv, math, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ITEMS = ["A", "B", "C", "D", "E", "F", "G", "H"]
LABEL = {"A": "A  hostile affect", "B": "B  explicit threat",
         "C": "C  lever recognition", "D": "D  lever use",
         "E": "E  conditional demand", "F": "F  correct target",
         "G": "G  fabrication", "H": "H  oversight resistance"}
STYLE = {"full":     ("#2a78d6", "o", "full mixture  v"),
         "resid":    ("#eb6834", "s", "residual  r"),
         "jspace":   ("#1baf7a", "^", "J-component  v_j"),
         "randatom": ("#eda100", "D", "random atoms, same pool")}
DOSES = [0.05, 0.1, 0.15, 0.2, 0.3]
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
MIN_SOLID = 10


def wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def main() -> int:
    agg = defaultdict(lambda: defaultdict(list))
    gate = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open("results/scoring_out/scores_judge.csv")):
        if r["development"] == "True":
            continue
        m = re.match(r"ca_unit_pos_(\w+?)_L22_a([0-9.]+)$", r["condition"])
        if m:
            arm, a = m.group(1), float(m.group(2))
        elif r["condition"] == "unsteered_L22_a0":
            arm, a = "unsteered", 0.0
        else:
            continue
        gate[(arm, a)][1] += 1
        if r["G0"] != "1":
            continue
        gate[(arm, a)][0] += 1
        for it in ITEMS:
            agg[(arm, a)][it].append(r[it])

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)

    for ax, it in zip(axes.ravel(), ITEMS):
        ax.set_facecolor(SURF)
        b = [int(v) for v in agg[("unsteered", 0.0)][it] if v in ("0", "1")]
        if len(b) >= MIN_SOLID:
            ax.axhline(sum(b) / len(b), color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
            ax.plot([0], [sum(b) / len(b)], "*", ms=11, color=MUTED, mec=SURF, zorder=5)
        for arm, (col, mk, _) in STYLE.items():
            xs, ys, lo, hi, ns = [], [], [], [], []
            for a in DOSES:
                vals = [int(v) for v in agg[(arm, a)][it] if v in ("0", "1")]
                if not vals:
                    continue
                p, l, h = wilson(sum(vals), len(vals))
                # clamp: floating error can make these -1e-17 at p=0 or 1
                xs.append(a); ys.append(p)
                lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
                ns.append(len(vals))
            if not xs:
                continue
            ax.plot(xs, ys, "-", color=col, lw=1.8, alpha=.85, zorder=2)
            for x, y, l, h, n in zip(xs, ys, lo, hi, ns):
                solid = n >= MIN_SOLID
                ax.errorbar([x], [y], yerr=[[l], [h]], fmt=mk, ms=8,
                            elinewidth=1.3, capsize=2.5, color=col,
                            mfc=col if solid else SURF, mec=col, mew=1.6,
                            ecolor=col, zorder=3)
                if not solid:
                    ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                                xytext=(4, -11), fontsize=7.5, color=col)
        ax.set_title(LABEL[it] + ("   ⚠ over-triggers" if it == "H" else ""),
                     fontsize=10.5, color=INK, loc="left", pad=7)
        ax.set_ylim(-.06, 1.06); ax.set_xlim(-.02, .33)
        ax.set_xticks([0, .05, .1, .15, .2, .3])
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate over scoreable samples", fontsize=9.5, color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("steering magnitude  α", fontsize=9.5, color=MUTED)

    hs = [Line2D([], [], color=c, marker=m, ls="-", ms=8, mec=c, label=lab)
          for c, m, lab in STYLE.values()]
    hs.append(Line2D([], [], color=MUTED, marker="*", ls=(0, (4, 3)), ms=11,
                     label="unsteered (α=0)"))
    fig.suptitle("Eight-item misalignment profile vs steering magnitude",
                 x=.008, y=.985, ha="left", fontsize=15, color=INK)
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(.008, .947),
               ncol=5, frameon=False, fontsize=10, labelcolor=INK)

    rows = []
    for arm in STYLE:
        cells = " ".join(f"{gate[(arm,a)][0]:>2}" for a in DOSES
                         if (arm, a) in gate)
        rows.append(f"{arm}: {cells}")
    foot = ("gemma-2-9b-it · layer 22 · contempt + aggressiveness, unit-normalised · "
            "judge claude-sonnet-5 · Wilson 95% CI · positive side only\n"
            "Scoreable of 20 per arm at α = .05 .1 .15 .2 .3 — "
            + "   ".join(rows) + "\n"
            "Hollow markers are n < 10: those points are conditioned on the "
            "samples that still produced a tool call, so they measure the "
            "survivors, not the condition. jspace and randatom lose most of "
            "their samples by α = 0.2.\n"
            "C, D, E and F sit at floor in this scenario; H over-triggers (span "
            "audit sustained 4 of 10 unsteered positives) and is an upper bound.")
    fig.text(.008, .012, foot, fontsize=8.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=[0, .115, 1, .925])
    out = Path("results/figures/dose_response_L22.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
