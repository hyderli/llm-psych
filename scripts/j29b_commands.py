"""Print the J29 (amendment 1) sweep commands from the build report.

Each arm runs at its own in-arm alpha, so each needs its own
run_dose_sweep.sh call; the condition name is <arm>_L16_a<alpha:.4f>.
Run the printed lines from emotion_steering/ on the pod.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rep = json.loads((ROOT / "results/j29/build_j29b_report.json").read_text())
base = "EPOCHS=40 LAYER=16 bash ../scripts/run_dose_sweep.sh"
print(f'ARMS="j24_full" DOSES="0" {base}')
print(f'ARMS="j24_pool1" DOSES="0.15" {base}')
print(f'ARMS="j25_coh1" DOSES="0.25" {base}')
for name, r in rep["arms"].items():
    print(f'ARMS="{r.get("file", name)}" DOSES="{r["alpha"]:.4f}" {base}')
axes = ROOT / "results/j29/build_j29b_axes_report.json"   # J29 amendment 2
if axes.exists():
    for name, r in json.loads(axes.read_text())["arms"].items():
        print(f'ARMS="{name}" DOSES="{r["alpha"]:.4f}" {base}')
