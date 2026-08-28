"""Apply the 2026-07-12 per-emotion C2 layer-selection rule and lock the result.

Implements the deterministic rule pre-registered in HYPOTHESES.md
(2026-07-12 amendment, "Per-emotion layer selection for C2 concept
validation"):

1. Per model x emotion, take implicit-scenario argmax accuracy at every
   layer of the sweep grid (``implicit_scenarios_sweep.md``).
2. Smooth with a 3-layer moving average (truncated at grid edges) and
   select the layer with maximum smoothed accuracy; exact ties (after
   rounding to 6 dp) break toward the shallower layer.
3. If the maximum *unsmoothed* accuracy is < 0.60, report "no
   recoverable layer" (vector-quality failure via step 3).
4. Vector-quality clause: if no layer of the intensity sweep
   (``intensity_semantic_sweep.md``) reaches inverse-family mean
   rho(rank) >= 0.6, the emotion is a vector-quality failure routed to
   residualization, regardless of implicit accuracy.
5. Loathing is grandfathered at the uniform ~2/3-depth convention layer
   (round(2/3 * n_layers), matching the headline reports).

Outputs (deterministic, no RNG):

- ``configs/vector_validation/layers.yaml``  — the locked layers.
- ``results/vector_validation/layer_selection_report.md`` — what was
  selected and why, including descriptive intensity rho at the selected
  layer (NOT confirmatory; confirmation requires the fresh MD5-frozen
  inverse families per the amendment).

Usage::

    uv run python scripts/select_c2_layers.py
    uv run python scripts/select_c2_layers.py --results-dir results/vector_validation

References
----------
HYPOTHESES.md, 2026-07-12 amendment; plans/layer-selection-amendment.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_repo_root = Path(__file__).resolve().parent.parent

EMOTIONS = ("admiration", "joy", "loathing", "sadness")
GRANDFATHERED = ("loathing",)  # passed all C2 tests at the convention layer
ACC_FLOOR = 0.60          # step 3: minimum unsmoothed accuracy
INTENSITY_TARGET = 0.60   # clause 4: inverse-family mean rho(rank) target
TIE_DECIMALS = 6          # exact-tie rounding for smoothed accuracies


@dataclass
class EmotionSelection:
    """Outcome of the rule for one model x emotion."""

    status: str                       # selected | grandfathered | vector_quality_failure | no_recoverable_layer
    layer: int | None = None
    smoothed_acc: float | None = None
    max_unsmoothed_acc: float | None = None
    intensity_rho_at_layer: float | None = None   # descriptive only
    max_intensity_rho: float | None = None        # over the sweep grid
    note: str = ""


@dataclass
class ModelSelection:
    model_key: str
    convention_layer: int
    emotions: dict[str, EmotionSelection] = field(default_factory=dict)


def parse_sweep_table(path: Path) -> dict[str, dict[int, float]]:
    """Parse a sweep markdown table into ``{column: {layer: value}}``.

    Expects a pipe table whose first column is ``layer``. Column names
    are taken from the header row; the ``neutral|rho|`` column (any
    header containing ``neutral``) is preserved under ``"neutral"``.

    Parameters
    ----------
    path
        Sweep report (``implicit_scenarios_sweep.md`` or
        ``intensity_semantic_sweep.md``).

    Returns
    -------
    dict
        Mapping of column name to ``{layer: value}``.
    """
    lines = path.read_text().splitlines()
    header: list[str] | None = None
    out: dict[str, dict[int, float]] = {}
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            if cells and cells[0].lower() == "layer":
                header = ["neutral" if "neutral" in c.lower() else c.lower() for c in cells]
                out = {c: {} for c in header[1:]}
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        try:
            layer = int(cells[0])
        except ValueError:
            continue
        for name, cell in zip(header[1:], cells[1:]):
            m = re.search(r"[-+]?\d*\.?\d+", cell)
            if m:
                out[name][layer] = float(m.group())
    if header is None:
        raise ValueError(f"no sweep table found in {path}")
    return out


def moving_average(series: dict[int, float]) -> dict[int, float]:
    """3-layer moving average, truncated at the ends of the layer grid.

    Edge layers average over the two layers available (self + one
    neighbor); interior layers over three. Deterministic; assumes the
    grid is contiguous (validated by the caller).
    """
    layers = sorted(series)
    smoothed: dict[int, float] = {}
    for i, layer in enumerate(layers):
        window = [series[layers[j]] for j in range(max(0, i - 1), min(len(layers), i + 2))]
        smoothed[layer] = sum(window) / len(window)
    return smoothed


def select_for_emotion(
    implicit: dict[int, float],
    intensity: dict[int, float],
) -> EmotionSelection:
    """Apply steps 1-4 of the rule for a single model x emotion."""
    max_rho = max(intensity.values())
    # Clause 4 first: construction failure is not a layer problem.
    if max_rho < INTENSITY_TARGET:
        return EmotionSelection(
            status="vector_quality_failure",
            max_intensity_rho=max_rho,
            note=(
                f"no layer reaches intensity inverse-family rho >= {INTENSITY_TARGET} "
                f"(max {max_rho:+.2f}); routed to residualization"
            ),
        )
    max_acc = max(implicit.values())
    if max_acc < ACC_FLOOR:  # step 3
        return EmotionSelection(
            status="no_recoverable_layer",
            max_unsmoothed_acc=max_acc,
            max_intensity_rho=max_rho,
            note=f"max implicit accuracy {max_acc:.2f} < {ACC_FLOOR}",
        )
    smoothed = moving_average(implicit)
    best = round(max(smoothed.values()), TIE_DECIMALS)
    layer = min(l for l, v in smoothed.items() if round(v, TIE_DECIMALS) == best)
    return EmotionSelection(
        status="selected",
        layer=layer,
        smoothed_acc=smoothed[layer],
        max_unsmoothed_acc=max_acc,
        intensity_rho_at_layer=intensity.get(layer),
        max_intensity_rho=max_rho,
    )


def convention_layer(n_layers: int) -> int:
    """Uniform ~2/3-depth convention layer, matching the headline reports."""
    return round(2 * n_layers / 3)


def model_configs() -> dict[str, int]:
    """Map HF model key (results dir name) -> n_layers from configs/model/."""
    out: dict[str, int] = {}
    for cfg in (_repo_root / "configs" / "model").glob("*.yaml"):
        data = yaml.safe_load(cfg.read_text())
        if not isinstance(data, dict) or data.get("n_layers") is None:
            continue  # e.g. rejected-model stubs without a full spec
        key = str(data["hf_model_id"]).split("/")[-1]
        out[key] = int(data["n_layers"])
    return out


def run_selection(results_dir: Path, models: list[str]) -> list[ModelSelection]:
    """Apply the full rule to every requested model."""
    n_layers_by_key = model_configs()
    selections: list[ModelSelection] = []
    for key in models:
        mdir = results_dir / key
        implicit = parse_sweep_table(mdir / "implicit_scenarios_sweep.md")
        intensity = parse_sweep_table(mdir / "intensity_semantic_sweep.md")
        conv = convention_layer(n_layers_by_key[key])
        ms = ModelSelection(model_key=key, convention_layer=conv)
        for emo in EMOTIONS:
            if emo in GRANDFATHERED:
                ms.emotions[emo] = EmotionSelection(
                    status="grandfathered",
                    layer=conv,
                    intensity_rho_at_layer=intensity[emo].get(conv),
                    note="passed all C2 tests at the convention layer (2026-07-12 amendment)",
                )
            else:
                ms.emotions[emo] = select_for_emotion(implicit[emo], intensity[emo])
        selections.append(ms)
    return selections


def write_yaml(selections: list[ModelSelection], out_path: Path) -> None:
    doc: dict = {
        "_locked": (
            "Per-emotion C2 validation layers, selected once by "
            "scripts/select_c2_layers.py per the HYPOTHESES.md 2026-07-12 "
            "amendment. Do not edit by hand; do not re-run after the "
            "confirmation stimuli exist."
        ),
        "rule": "HYPOTHESES.md 2026-07-12 amendment (per-emotion C2 layer selection)",
        "models": {},
    }
    for ms in selections:
        doc["models"][ms.model_key] = {
            "convention_layer": ms.convention_layer,
            "emotions": {
                emo: {
                    "status": sel.status,
                    "layer": sel.layer,
                    **({"smoothed_acc": round(sel.smoothed_acc, 4)} if sel.smoothed_acc is not None else {}),
                    **({"note": sel.note} if sel.note else {}),
                }
                for emo, sel in ms.emotions.items()
            },
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, width=78))


def write_report(selections: list[ModelSelection], out_path: Path) -> None:
    lines = [
        "# C2 per-emotion layer selection — rule output",
        "",
        "- Rule: HYPOTHESES.md **2026-07-12 amendment** (implicit-accuracy 3-layer",
        "  moving average, shallow tie-break, vector-quality clause).",
        "- Inputs: `implicit_scenarios_sweep.md` / `intensity_semantic_sweep.md` per model.",
        "- The intensity rho shown at the selected layer is **descriptive only** —",
        "  already-observed sweep data. Confirmatory status requires the fresh",
        "  MD5-frozen inverse families at the locked layers.",
        "",
        "| model | emotion | status | layer | smoothed acc | rho(rank) at layer | max sweep rho |",
        "|---|---|---|---|---|---|---|",
    ]
    for ms in selections:
        for emo, sel in ms.emotions.items():
            lines.append(
                f"| {ms.model_key} | {emo} | {sel.status} | "
                f"{sel.layer if sel.layer is not None else '—'} | "
                f"{f'{sel.smoothed_acc:.3f}' if sel.smoothed_acc is not None else '—'} | "
                f"{f'{sel.intensity_rho_at_layer:+.2f}' if sel.intensity_rho_at_layer is not None else '—'} | "
                f"{f'{sel.max_intensity_rho:+.2f}' if sel.max_intensity_rho is not None else '—'} |"
            )
    lines += [
        "",
        "## Notes",
        "",
    ]
    for ms in selections:
        for emo, sel in ms.emotions.items():
            if sel.note:
                lines.append(f"- **{ms.model_key} / {emo}:** {sel.note}")
    lines += [
        "",
        "## Next steps (per the amendment)",
        "",
        "1. Author >=3 NEW inverse families per surviving emotion; freeze MD5 in",
        "   `configs/stimuli_hashes.yaml` BEFORE any model run.",
        "2. Run the intensity confirmation at the locked layers only.",
        "3. Vector-quality failures -> `plans/residualization-admiration.md`.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path,
                    default=_repo_root / "results" / "vector_validation")
    ap.add_argument("--models", nargs="+",
                    default=["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"],
                    help="results-dir subfolders (HF model keys) to process")
    ap.add_argument("--out-yaml", type=Path,
                    default=_repo_root / "configs" / "vector_validation" / "layers.yaml")
    ap.add_argument("--out-report", type=Path,
                    default=_repo_root / "results" / "vector_validation" / "layer_selection_report.md")
    args = ap.parse_args()

    if args.out_yaml.exists():
        print(f"REFUSING to overwrite existing lock file {args.out_yaml}\n"
              f"(the amendment allows one application of the rule; delete it "
              f"manually only if the selection has never been used).", file=sys.stderr)
        return 1

    selections = run_selection(args.results_dir, args.models)
    write_yaml(selections, args.out_yaml)
    write_report(selections, args.out_report)
    print(f"locked  -> {args.out_yaml.relative_to(_repo_root)}")
    print(f"report  -> {args.out_report.relative_to(_repo_root)}")
    for ms in selections:
        for emo, sel in ms.emotions.items():
            print(f"  {ms.model_key:26s} {emo:11s} {sel.status:24s} "
                  f"layer={sel.layer if sel.layer is not None else '—'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
