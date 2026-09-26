"""Item rates for the contempt+aggression arms, per eight-item blackmail score.

x is SIGNED steering magnitude: the arm vectors carry their sign, so the negative
arms plot at negative alpha and unsteered sits at 0. That puts both signs on one
axis honestly -- it is one dose axis through the contempt+aggression direction.

What this figure is NOT: a dose-response curve. Each arm has one usable alpha
(jspace at 0.3 was gate-flagged 20/20, so no scores exist; randatom at +0.3 has
n=1), so points are drawn WITHOUT connecting lines. Joining single points would
invent a trend.

Rates are over coherent, non-development samples, matching
`score_blackmail.py combine`. Error bars are Wilson 95% intervals. Cells with
n < MIN_N are omitted and listed in the footnote rather than drawn, since a
one-sample rate reads as a measurement.
"""
from __future__ import annotations

import csv
import math
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
ARMS = {"ca_unit_pos_full": ("full", +1), "ca_unit_neg_full": ("full", -1),
        "ca_unit_pos_jspace": ("jspace", +1), "ca_unit_neg_jspace": ("jspace", -1),
        "ca_unit_pos_randatom": ("randatom", +1),
        "ca_unit_neg_randatom": ("randatom", -1),
        "unsteered": ("unsteered", 0)}
MIN_N = 5

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light).
# Marker shape is the secondary encoding the aqua slot's sub-3:1 contrast requires.
STYLE = {"full":     ("#2a78d6", "o", "full mixture"),
         "jspace":   ("#eb6834", "s", "J-component"),
         "randatom": ("#1baf7a", "^", "random lens atoms")}
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def main() -> int:
    src = Path("results/scoring_out/scores_judge.csv")
    agg: dict = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(src.open()):
        if r["development"] == "True" or r["G0"] != "1":
            continue                       # matches combine's exclusions
        stem, _, atail = r["condition"].rpartition("_L22_a")
        stem = stem or "unsteered"
        if stem not in ARMS:
            continue
        arm, sign = ARMS[stem]
        a = 0.0 if arm == "unsteered" else sign * float(atail)
        for it in ITEMS:
            v = r[it]
            agg[(arm, a)][it].append(int(v) if v in ("0", "1") else None)

    base = {it: wilson(sum(v for v in agg[("unsteered", 0.0)][it] if v is not None),
                       len([v for v in agg[("unsteered", 0.0)][it] if v is not None]))
            for it in ITEMS}

    doses = sorted({a for (_, a) in agg})
    xpos = {a: i for i, a in enumerate(doses)}

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)
    omitted, na_note = set(), []

    for ax, it in zip(axes.ravel(), ITEMS):
        ax.set_facecolor(SURF)
        b_n = len([v for v in agg[("unsteered", 0.0)][it] if v is not None])
        if b_n >= MIN_N:
            ax.axhline(base[it][0], color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
        ax.axvline(xpos[0.0], color=GRID, lw=1, zorder=0)
        for di, (arm, (col, mk, _)) in enumerate(STYLE.items()):
            dx = (di - 1) * 0.15       # legibility dodge only; true α on ticks
            xs, ys, los, his = [], [], [], []
            for (a_arm, a), d in sorted(agg.items(), key=lambda x: x[0][1]):
                if a_arm != arm:
                    continue
                vals = [v for v in d[it] if v is not None]
                n_na = sum(1 for v in d[it] if v is None)
                if it == "F" and n_na:
                    na_note.append((arm, a, n_na))
                if len(vals) < MIN_N:
                    # n == 0 on F means "not applicable", not "too few"
                    if len(vals) and it != "F":
                        omitted.add((arm, a, len(vals)))
                    continue
                p, lo, hi = wilson(sum(vals), len(vals))
                xs.append(xpos[a] + dx); ys.append(p); los.append(p - lo)
                his.append(hi - p)
            if xs:
                ax.errorbar(xs, ys, yerr=[los, his], fmt=mk, ms=9, lw=0,
                            elinewidth=1.6, capsize=3, color=col, mec=SURF,
                            mew=1.2, ecolor=col, zorder=3)
        # unsteered baseline marker
        if b_n >= MIN_N:
            ax.plot([xpos[0.0]], [base[it][0]], "D", ms=8, color=MUTED, mec=SURF, mew=1.2,
                    zorder=4)
        else:
            ax.text(0.5, 0.93, f"baseline n={b_n}, not drawn", fontsize=8,
                    color=MUTED, ha="center", transform=ax.transAxes)
        ttl = LABEL[it]
        if it == "H":
            ttl += "   ⚠ over-triggers"
        if it == "F":
            ttl += "   (where applicable)"
        ax.set_title(ttl, fontsize=10.5, color=INK, loc="left", pad=7)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xlim(-0.6, len(doses) - 0.4)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([("0" if a == 0 else f"{a:+.2f}") for a in doses])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)

    for ax in axes[:, 0]:
        ax.set_ylabel("rate over coherent samples", fontsize=9.5, color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("signed steering magnitude  α", fontsize=9.5, color=MUTED)

    handles = [Line2D([], [], color=c, marker=m, ls="", ms=9, mec=SURF, label=lab)
               for c, m, lab in STYLE.values()]
    handles.append(Line2D([], [], color=MUTED, marker="D", ls=(0, (4, 3)), ms=8,
                          label="unsteered (α=0), dashed = its rate"))
    fig.suptitle("Eight-item misalignment profile by signed steering magnitude",
                 x=0.008, y=0.985, ha="left", fontsize=15, color=INK)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.945),
               ncol=4, frameon=False, fontsize=10, labelcolor=INK)

    import textwrap
    om = (", ".join(f"{a}@{x:+.2f} (n={n})" for a, x, n in sorted(omitted))
          or "none")
    foot = ("gemma-2-9b-it · layer 22 · contempt + aggressiveness, unit-normalised · "
            "judge claude-sonnet-5 · Wilson 95% CI · rates exclude gate-flagged and "
            "development samples\n"
            "Points are NOT joined: each arm has one usable α, so no trend is implied. "
            "x is ORDINAL — the six doses that exist, equally spaced — not a "
            "linear dose axis; series are dodged sideways for legibility only. "
            f"Omitted for n < {MIN_N}: " + om + ". "
            "jspace at |α|=0.3 has no scoreable samples at all — gate-flagged "
            "20/20 on both signs, so its panel positions are simply empty.\n"
            "H over-triggers: a span audit sustained 4 of 10 unsteered positives "
            "outright and found one item-boundary error (lever recognition scored as "
            "H). Read its panel as an upper bound.")
    foot = "\n".join(textwrap.fill(ln, 155) for ln in foot.split("\n"))
    fig.text(0.008, 0.012, foot, fontsize=8.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=[0, 0.12, 1, 0.915])
    out = Path("results/figures/item_rates_L22.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(f"-> {out}")
    if na_note:
        print("F not-applicable counts (B=0):",
              ", ".join(f"{a}@{x:+.2f}:{n}" for a, x, n in sorted(set(na_note))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
