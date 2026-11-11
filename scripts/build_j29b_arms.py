"""J29 amendment 1 arm builder (tag j29b). Standard library only.

Rebuilds the J29 split without the emotion-vector leak, adds coh splits and
random-in-G controls, and records every arm's in-arm alpha. Fixed in
plans/j-space-decomposition.md ("J29 amendment 1"), before any J29 sample.

For a source arm a (unit), v_hat = unit(j24_full), w = a - (a.v_hat) v_hat:

    g' = P_G w - (v_hat . P_G w) v_hat      (G part, emotion-vector removed)
    s' = w - g'                              (remainder; perpendicular to v_hat)

g' + s' = w exactly. In-arm alpha of a component = base_alpha * |component|,
because the steering hook normalises every direction and injects
alpha * unit(d) * |h| (J25 amendment 1, point 5).

Families (6 each):
    poolG / poolS   from j24_pool{i}, base alpha 0.15
    cohG  / cohS    from j25_coh{i},  base alpha 0.25 (coh's measured dose)
    grand           random unit directions inside G and perpendicular to
                    v_hat; draw i runs at poolG{i}'s alpha
    poolSm          poolS{i} at poolG{i}'s alpha (matched-dose framing)

G is the stored rank-32 basis results/j27/subspaces_clean/_G/G_l16.npy.

Usage::

    python3 scripts/build_j29b_arms.py            # report only
    python3 scripts/build_j29b_arms.py --write    # report + .npy files
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEC = ROOT / "steering_vectors" / "Qwen2.5-7B-Instruct-story-wheel32"
G_PATH = ROOT / "results" / "j27" / "subspaces_clean" / "_G" / "G_l16.npy"
REPORT = ROOT / "results" / "j29" / "build_j29b_report.json"

# --- frozen by J29 amendment 1 ---------------------------------------------
SEED = 20261112
N = 6
BASE = {"pool": 0.15, "coh": 0.25}
SRC = {"pool": "j24_pool{}", "coh": "j25_coh{}"}
TAG = "j29b"
TOL_V = 1e-6          # |cos(component, v_hat)| must be below this
TOL_SUM = 1e-6        # |g' + s' - w| / |w| must be below this
# ---------------------------------------------------------------------------


def load_npy(path: Path) -> tuple[list[float], dict]:
    b = path.read_bytes()
    if b[:6] != b"\x93NUMPY":
        raise SystemExit(f"not an npy file: {path}")
    major = b[6]
    if major == 1:
        hl, off = struct.unpack("<H", b[8:10])[0], 10
    else:
        hl, off = struct.unpack("<I", b[8:12])[0], 12
    h = ast.literal_eval(b[off:off + hl].decode("latin1"))
    code = {"<f4": "f", "<f8": "d"}[h["descr"]]
    data = b[off + hl:]
    n = len(data) // struct.calcsize(code)
    return list(struct.unpack(f"<{n}{code}", data)), h


def save_npy(path: Path, x: list[float]) -> None:
    hdr = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({len(x)},), }}"
    pad = 64 - (10 + len(hdr) + 1) % 64
    hdr = hdr + " " * pad + "\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr))
                     + hdr.encode("latin1") + struct.pack(f"<{len(x)}f", *x))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a):
    return math.sqrt(dot(a, a))


def scale(a, c):
    return [c * x for x in a]


def sub(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def add(a, b):
    return [x + y for x, y in zip(a, b, strict=True)]


def cos(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    flat, h = load_npy(G_PATH)
    d, k = h["shape"]
    if h["fortran_order"]:
        cols = [flat[j * d:(j + 1) * d] for j in range(k)]
    else:
        cols = [[flat[i * k + j] for i in range(d)] for j in range(k)]

    def proj_g(x):
        c = [dot(col, x) for col in cols]
        out = [0.0] * d
        for j in range(k):
            cj, col = c[j], cols[j]
            for i in range(d):
                out[i] += cj * col[i]
        return out

    full, _ = load_npy(VEC / "j24_full_layer16.npy")
    target = norm(full)
    vh = scale(full, 1 / target)
    pv = proj_g(vh)                     # v_hat's part inside G

    arms: dict[str, list[float]] = {}
    rep: dict = {"amendment": "J29 amendment 1", "seed": SEED, "G": str(G_PATH.relative_to(ROOT)),
                 "G_rank": k, "v_hat_share_in_G": dot(pv, pv), "arms": {}, "families": {}}

    def record(name, x, alpha, extra):
        rep["arms"][name] = {"alpha": round(alpha, 4), "cos_v": cos(x, vh),
                             "share_in_G": dot(proj_g(x), proj_g(x)) / dot(x, x), **extra}
        if abs(rep["arms"][name]["cos_v"]) > TOL_V:
            raise SystemExit(f"REFUSING {name}: cos with v_hat {rep['arms'][name]['cos_v']:.2e}")
        arms[name] = x

    gparts: dict[str, list] = {"pool": [], "coh": []}
    for fam in ("pool", "coh"):
        for i in range(1, N + 1):
            a, _ = load_npy(VEC / f"{SRC[fam].format(i)}_layer16.npy")
            a = scale(a, 1 / norm(a))
            w = sub(a, scale(vh, dot(a, vh)))
            pg = proj_g(w)
            g = sub(pg, scale(vh, dot(vh, pg)))
            s = sub(w, g)
            err = norm(sub(add(g, s), w)) / norm(w)
            if err > TOL_SUM:
                raise SystemExit(f"REFUSING {fam}{i}: split does not sum to w ({err:.2e})")
            base = BASE[fam]
            src = SRC[fam].format(i)
            record(f"{TAG}_{fam}G{i}", g, base * norm(g),
                   {"source": src, "norm_in_arm": norm(g), "cos_with_S": cos(g, s)})
            record(f"{TAG}_{fam}S{i}", s, base * norm(s),
                   {"source": src, "norm_in_arm": norm(s)})
            gparts[fam].append(g)
            if fam == "pool":
                rep["arms"][f"{TAG}_poolSm{i}"] = {**rep["arms"][f"{TAG}_poolS{i}"],
                                                   "alpha": round(base * norm(g), 4),
                                                   "file": f"{TAG}_poolS{i}",
                                                   "framing": "matched to poolG alpha"}

    # random directions inside G, perpendicular to v_hat (stay inside G by
    # removing their component along P_G v_hat, which lies in G)
    rng = random.Random(SEED)
    pvn = dot(pv, pv)
    for i in range(1, N + 1):
        c = [rng.gauss(0.0, 1.0) for _ in range(k)]
        x = [0.0] * d
        for j in range(k):
            for r in range(d):
                x[r] += c[j] * cols[j][r]
        x = sub(x, scale(pv, dot(x, pv) / pvn))
        alpha = rep["arms"][f"{TAG}_poolG{i}"]["alpha"]
        record(f"{TAG}_grand{i}", x, alpha, {"matched_to": f"{TAG}_poolG{i}"})

    def within(vs):
        cs = [cos(vs[a], vs[b]) for a in range(len(vs)) for b in range(a + 1, len(vs))]
        return {"mean": sum(cs) / len(cs), "min": min(cs), "max": max(cs)}

    grand = [arms[f"{TAG}_grand{i}"] for i in range(1, N + 1)]
    rep["families"] = {
        "poolG_within": within(gparts["pool"]),
        "cohG_within": within(gparts["coh"]),
        "grand_within": within(grand),
        "poolG_vs_cohG_mean": sum(cos(a, b) for a in gparts["pool"] for b in gparts["coh"]) / N**2,
        "poolG_vs_grand_mean": sum(cos(a, b) for a in gparts["pool"] for b in grand) / N**2,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2))
    print(f"v_hat share inside G: {rep['v_hat_share_in_G']:.4f}")
    for n, r in rep["arms"].items():
        print(f"{n:16s} alpha {r['alpha']:.4f}  cos_v {r['cos_v']:+.1e}  "
              f"in G {r['share_in_G']:.3f}" + (f"  cos(G,S) {r['cos_with_S']:+.3f}"
                                               if "cos_with_S" in r else ""))
    print(json.dumps(rep["families"], indent=2))
    print(f"report -> {REPORT.relative_to(ROOT)}")
    if args.write:
        for n, x in arms.items():
            out = VEC / f"{n}_layer16.npy"
            if out.exists():
                raise SystemExit(f"REFUSING to overwrite {out}")
            save_npy(out, scale(x, target / norm(x)))
        print(f"{len(arms)} arm files -> {VEC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
