"""Build the human rating sheet for the C2 wheel32 stimulus set.

Purpose (plans/c2-revalidation-wheel32.md, A4): the complement-pair design is
robust to unequal ranges, unequal slopes and framing asymmetry, but NOT to an
arm that fails to be monotone -- that depresses |rho| in a way indistinguishable
from a weak probe. This sheet tests exactly that, and records a per-arm
attenuation ceiling to read probe scores against.

Design choices that matter, and why:

* **Order is shuffled and metadata is hidden.** If a family's rows appear
  together in x order, the rater is transcribing design intent rather than
  reading sentences. Item ids are opaque and the key is written to a separate
  file that must not be open while rating.
* **Both constructs are rated on every item.** Asking only the target construct
  would reveal, item by item, which construct an item belongs to. Rating both
  keeps the question uniform and yields a discriminant check for free: a
  desperation item should sit high on desperate and lower on nervous.
* **Five x values per arm.** Monotonicity is a shape question and does not need
  the full grid. Decreasing arms take {1,3,6,9,12}; increasing arms take the
  complementary values {19,17,14,11,8} so both sample the same underlying k.
* **Duplicated items.** With a single rater the only available reliability
  estimate is test-retest, so ~15% of items appear twice under different ids.

Deterministic: seeded, so the sheet can be regenerated and the key recovered.

Run:  uv run python scripts/build_rating_sheet.py
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
OUT_DIR = _repo_root / "data" / "derived" / "rating"
SHEET = OUT_DIR / "rating_sheet.csv"
KEY = OUT_DIR / "rating_key.csv"

SEED = 20260910
DEC_XS = [1, 3, 6, 9, 12]
INC_XS = [20 - x for x in DEC_XS]          # 19, 17, 14, 11, 8
NEUTRAL_XS = [1, 5, 10, 15, 19]
DUPLICATE_FRACTION = 0.15

# (emotion, scenario, family, direction, template)
FAMILIES = [
    # --- confirmation set, drafted 2026-09-10 ---
    ("desperation", "coal", "coal_remaining", "decreasing",
     "Of the twenty bags of coal delivered, she still has {x} of them in the shed."),
    ("desperation", "coal", "coal_burned", "increasing",
     "Of the twenty bags of coal delivered, she has already burned {x} of them."),
    ("neutral", "coal", "coal_lorry", "flat",
     "Of the twenty bags of coal delivered, the merchant brought {x} of them on the second lorry."),
    ("desperation", "childcare", "nights_covered", "decreasing",
     "Of the twenty nights she needs someone to watch her son, she has someone lined up for {x} of them."),
    ("desperation", "childcare", "nights_seeking", "increasing",
     "Of the twenty nights she needs someone to watch her son, she is still looking for someone for {x} of them."),
    ("neutral", "childcare", "nights_phone", "flat",
     "Of the twenty nights she needs someone to watch her son, she has {x} of them written in her phone rather than on the wall calendar."),
    ("desperation", "arrears", "payments_held", "decreasing",
     "The agency has ruled on all twenty missed payments on her file; it agreed to hold {x} of them."),
    ("desperation", "arrears", "payments_pursued", "increasing",
     "The agency has ruled on all twenty missed payments on her file; it is still pursuing {x} of them."),
    ("neutral", "arrears", "payments_prefix", "flat",
     "The agency has ruled on all twenty missed payments on her file; it lists {x} of them under the same reference prefix."),
    ("nervousness", "gathering", "faces_known", "decreasing",
     "Of the twenty people in the room tonight, she has met {x} of them before."),
    ("nervousness", "gathering", "faces_new", "increasing",
     "Of the twenty people in the room tonight, she is meeting {x} of them for the first time."),
    ("neutral", "gathering", "replies_sameday", "flat",
     "Of the twenty people in the room tonight, the host heard back from {x} of them on the same day."),
    ("nervousness", "inspection", "entries_cleared", "decreasing",
     "Of the twenty entries the inspector has ruled on, he cleared {x} of them."),
    ("nervousness", "inspection", "entries_violations", "increasing",
     "Of the twenty entries the inspector has ruled on, he recorded {x} of them as violations."),
    ("neutral", "inspection", "entries_onehand", "flat",
     "Of the twenty entries the inspector has ruled on, one clerk wrote {x} of them."),
    ("nervousness", "pumps", "pumps_steady", "decreasing",
     "Of the twenty pumps on the floor tonight, she has watched {x} of them run steadily."),
    ("nervousness", "pumps", "pumps_cutout", "increasing",
     "Of the twenty pumps on the floor tonight, she has watched {x} of them cut out at least once."),
    ("neutral", "pumps", "pumps_plate", "flat",
     "Of the twenty pumps on the floor tonight, the maintenance file lists {x} of them with the serial plate on the housing rather than the base."),
]

INSTRUCTIONS = [
    "# C2 wheel32 rating sheet",
    "#",
    "# For each sentence, rate the person described in it on BOTH scales.",
    "# Judge only what the sentence says. Do not try to be consistent with",
    "# earlier items -- the order is deliberately shuffled and some sentences",
    "# appear more than once.",
    "#",
    "#   0  = not at all      3 = slightly      5 = moderately",
    "#   7  = considerably   10 = extremely",
    "#",
    "# desperate = high need with few options left",
    "# nervous   = anxious about something that has not happened yet",
    "#",
    "# Fill the 'desperate' and 'nervous' columns with a whole number 0-10.",
    "# Leave nothing blank. Do NOT open rating_key.csv until you are finished.",
    "#",
]


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    items: list[dict] = []
    for emotion, scenario, family, direction, template in FAMILIES:
        xs = {"decreasing": DEC_XS, "increasing": INC_XS, "flat": NEUTRAL_XS}[direction]
        for x in xs:
            items.append({
                "emotion": emotion, "scenario": scenario, "family": family,
                "direction": direction, "x": x, "text": template.format(x=x),
            })

    n_dup = round(len(items) * DUPLICATE_FRACTION)
    dups = [dict(it, is_duplicate=True) for it in rng.sample(items, n_dup)]
    for it in items:
        it.setdefault("is_duplicate", False)
    allitems = items + dups
    rng.shuffle(allitems)

    for i, it in enumerate(allitems, start=1):
        h = hashlib.md5(f"{SEED}:{i}:{it['family']}:{it['x']}".encode()).hexdigest()[:6]
        it["item_id"] = f"it{i:03d}{h}"

    sheet = [{"item_id": it["item_id"], "sentence": it["text"],
              "desperate": "", "nervous": ""} for it in allitems]
    key = [{"item_id": it["item_id"], "emotion": it["emotion"],
            "scenario": it["scenario"], "family": it["family"],
            "direction": it["direction"], "x": it["x"],
            "is_duplicate": int(it["is_duplicate"])} for it in allitems]
    return sheet, key


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet, key = build()

    with SHEET.open("w", newline="") as f:
        for line in INSTRUCTIONS:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=["item_id", "sentence", "desperate", "nervous"])
        w.writeheader()
        w.writerows(sheet)

    with KEY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "emotion", "scenario",
                                          "family", "direction", "x", "is_duplicate"])
        w.writeheader()
        w.writerows(key)

    n_unique = sum(1 for r in key if not int(r["is_duplicate"]))
    print(f"wrote {SHEET}  ({len(sheet)} items, {n_unique} unique, "
          f"{len(sheet) - n_unique} repeats)")
    print(f"wrote {KEY}   -- do not open until rating is finished")


if __name__ == "__main__":
    main()
