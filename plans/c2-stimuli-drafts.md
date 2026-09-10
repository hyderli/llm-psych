# C2 wheel32 stimulus drafts — awaiting audit

**Status:** DRAFT, not frozen, not built, not hashed. Nothing here has been
audited by the PI. No file in `data/public/` has been touched.

**Date drafted:** 2026-09-08 (LLM-drafted, Claude), for the Tier A set resolved
in `plans/c2-revalidation-wheel32.md` (D1).

Provenance note: as with the July confirmation set, these are LLM-drafted
against the frozen constraints and require PI audit before freezing. That
deviation from "hand-authored" is already disclosed in the 2026-07-28
amendment block and applies here identically.

---

## 1. Sweep families — audited 2026-09-09, revised

PI audit rejected four of the six drafts. Recorded here with reasons, because
each failure mode recurs elsewhere in the set.

### Authoring rule adopted from the audit

A usable inverse family counts **remaining options or coverage**, is **bounded
by a stated denominator**, and stays **monotone at both endpoints, including
x = 0**. Counting elapsed or remaining *time* fails the first clause: it varies
the "when" and leaves the option set untouched.

Time-pressure is the specific danger. It is present in desperation,
nervousness, fear, apprehension and dread alike, so a family that loads on it
cannot be diagnostic for any single cell — it contaminates the between-category
comparisons, not just its own item.

### Rejected

| family | emotion | reason |
|---|---|---|
| `notice_days` | desperation | Measures urgency, not desperation. Desperation is high need under *shrinking options*; a deadline count leaves the option set fixed. 3 days with a sister's spare room is not desperate; 60 days with nowhere to go is. |
| `rent_months` | desperation | The same time construct as `notice_days` in different clothes — financial runway. Not independent evidence. |
| `runthroughs_done` | nervousness | Backwards at high x. "Run through the talk 30 times already this week" reads as anxious over-rehearsal, not preparedness; "already this week" implies compulsion. The function is U-shaped, and a U is indistinguishable from probe failure. |
| `weeks_notice` | nervousness | Time again, and directionally ambiguous: more notice means more preparation but also more anticipatory dread, and a long lead time signals the panel matters. |

`rooms_offered` survived the audit as the only draft that counts options
directly, but is superseded below by a bounded version.

### Second audit, 2026-09-09 — design rules added

**All-inverse is itself a confound.** The June/July rationale for excluding
increasing families ("for them rho(rank) == rho(x) identically, so they cannot
separate meaning from digit") is correct about a family *in isolation* and
wrong about the set. If every family decreases with x, a probe that reads
numeral magnitude with a negative weight scores rho(rank) = +1 on all of them,
with no emotional content whatsoever — six clean, consistent,
cross-family-generalizing results that mean nothing. The neutral families are
currently the only thing standing between that reading and the emotional one,
and they are three impersonal one-off scenarios, which is a thin subtraction.

The fix is inside the items already: the logical complement of each sentence.
Same scenario, same denominator, same numerals, same grammatical frame,
opposite predicate. Magnitude cannot flip sign when only the predicate changes,
so a probe that tracks x downward in the original and upward in the complement
has ruled magnitude out — a within-scenario control, far tighter than the
between-scenario neutral families.

Rule adopted: **at least one complement family per emotion, plus one
numeral-matched neutral in the same frame.** Increasing families are admitted
only as complements; the diagnostic unit is the pair's sign difference, not
either family's rho alone. This amends the July exclusion and must be disclosed
as such.

**Count position.** The count sits in an object or partitive-object position in
every family, so it never governs a verb. This is what makes x = 1 safe without
per-template rewording, and it is required by
`test_template_holds_structure_constant`: a frame where x = 1 needs "has" and
x = 2 needs "have" produces two skeletons and fails the test outright. Skipping
x = 1 was the wrong fix — 0 → 1 → 2 is the highest-curvature region in a
bounded-coverage family (one shelter bed is categorically unlike none, and
unlike two), so deleting it to dodge an agreement error removes the most
informative point in the family.

**Denominators** are spelled as words, and sit strictly above max(x) with
margin. A denominator equal to max(x) makes the top of the range exact complete
coverage, which is a regime change rather than a point on a continuum, and
reads unnaturally ("answers ready for 12 of them" where a person writes "all of
them").

**Per-family grids.** Value grids are set per family against that family's
saturation profile — dense where the curve is steep, sparse where it is flat —
rather than shared. Rank correlation is the common currency across families, so
unequal spacing costs nothing, and this dissolves the unbounded-family problem
instead of working around it. C2 scores Spearman, so nothing downstream assumes
equal spacing; do not fit a linear model on `intensity_rank`.

### Numeral tokenization — partly resolved, no longer blocking

`scripts/check_numeral_tokenization.py` still needs to run on a machine with
model access, but it no longer gates authoring, for two reasons.

**Every denominator is now twenty**, which exceeds max(x) under both the
thirteen-point and the ten-point grid. The grid decision no longer touches
coherence, so the families can be authored before the check runs.

**The readout position makes the length shift harmless here.**
`scripts/validate_intensity_semantic.py:162` records with
`token_position="last"`, and `src/llm_psych/hooks.py:78` implements that as
`hidden_states[:, -1, :]`. A numeral that splits into two tokens shifts every
following position by one, but the final position is invariant, so the C2
intensity test cannot pick up a step discontinuity from tokenization.

This is **not** true everywhere in the repo: `src/llm_psych/simple_probe.py`
pools at token 50, a fixed absolute index. Anything reading through that path
does have the exposure, and the script now prints sentence length per x, plus
the token count of each spelled-out denominator, so the exposure can be
measured rather than assumed.

What remains at stake in the check is only the grid size: if 10–12 split, capping
at 9 gives n = 10 per family.

### N_PER_FAMILY is stale regardless

Recovering x = 1 makes the uncapped grid {0…12}, which is thirteen values, so
`N_PER_FAMILY = 12` is already wrong independent of the tokenizer outcome. It
becomes 13 (uncapped) or 10 (capped). Either way it is a disclosed change to a
pre-registered constant, not a silent edit.

### The pair is a two-part statistic, not a sign check

- an emotion-reading probe is **antisymmetric** across the pair, since the
  complement inverts intensity while holding the scenario fixed;
- a magnitude-reading probe is **symmetric**, since magnitude is blind to the
  predicate.

Pre-register `(rho_A - rho_B) / 2` as the primary statistic and
`(rho_A + rho_B) / 2` as the artifact estimate, with **A always the decreasing
arm** — the antisymmetric half's sign depends on that assignment, so without
pinning it half the pairs report +0.7 and half -0.7 for identical results.

**Convention, and it inverts if got wrong.** Computed on `rho_x`, never
`rho_rank`. `intensity_rank` decreases with x in an inverse family and
increases with x in its complement, so `rho_rank = -rho_x` on one side of the
pair and `+rho_x` on the other, and on `rho_rank` the two halves swap roles
exactly: emotion lands in the symmetric half, magnitude in the antisymmetric
one. Both already exist at `scripts/validate_intensity_semantic.py:203-204`;
the gate uses `rho_rank`, so the pair analysis must reach past it.

**Assert the relationship rather than trusting it.** For every family,
`rho_rank == -rho_x` when direction is decreasing and `+rho_x` when increasing.
Two lines, and it catches a flipped `direction` field or a complement
registered with the wrong one — the failure most likely to survive review,
because the numbers stay plausible either way.

### The symmetric half conflates two things

Spearman is scale-free, so unequal intensity slopes between arms do not bias
rho when the relation is clean. What they do is **attenuate the arms
unequally**. A pair at |rho| 0.9 and 0.5 yields an antisymmetric half of 0.7
and a symmetric half of 0.2 with no numeral magnitude in it at all. So the
symmetric half carries both genuine magnitude contamination and any difference
in how cleanly the two arms encode intensity, and anything that makes one arm
mushier than its partner reads out as contamination.

Two consequences, both binding.

**Complements must partition the denominator.** A pair whose counts do not sum
to the denominator has a slack category, which muddies one arm and puts a false
signal into the artifact estimate. Three of the second-pass pairs were loose:
a shelter she called may have neither offered nor refused; a published question
may be neither answered nor actively in progress; a place may have gone quiet
without rejecting her. All three are fixed below by a predicate change plus, for
the shelters, "reached" in place of "called" so the set is exhaustive.

**Six neutrals, one per scenario.** Two neutrals covering two of six scenarios
cannot separate the two contributions in the other four. If a pair's symmetric
half is 0.2 and that scenario's neutral is flat, the 0.2 is attenuation and the
pair is fine; if the neutral moves with x, it is magnitude and the pair is
compromised. Those call for opposite responses. Scenario-matched neutrals cost
authoring only and convert the symmetric half from a number to argue about into
one that decomposes.

**Lexically positive, not morphologically negative.** `unprepared`, `unanswered`
and the like are unsafe substitutes for a positive predicate, since a
negation-sensitive probe may key on the affix. Blank, full, missing, out of the
running are all lexically positive.

### Exactness forces complementary grids

If arm A counts k and arm B counts twenty minus k, running the *same* x-grid on
both makes them sample **different regions of the same underlying variable**.
Desperation is steep near k = 0 and flat at large k, so one arm would sit on the
steep part and the other on the flat part, the second arm's |rho| would
attenuate for that reason alone, and the difference would land in the symmetric
half — precisely the false signal the partition rule exists to remove.

**Rule: the grid is closed under the complement map.**

- decreasing arms: x in {1…12}
- increasing arms: x in {8…19}
- neutrals: x in {1…19}

Both arms then sample k in 1…12: the same world states, the same region of the
curve. Twelve values per arm, so `N_PER_FAMILY = 12` survives intact after all.

Neutrals take the **union** of the two arms' numerals, not either arm's. A
neutral run only over 1…12 would test magnitude in A's range and say nothing
about 13…19, where B does half its sampling. The union also gives each neutral
more points than either arm, which is what is wanted from an instrument whose
job is to certify a null.

**k = 0 is dropped, and not for the phrasing.** It is a categorical state
rather than the bottom of a continuum — nobody left to ask, nowhere still open,
nothing in the folder — which is the same objection that retired `prior_talks`
at x = 0 and the same one that put every denominator above max(x). The awkward
rendering at both ends ("still in the running at 0", "out of the running at 20
of the twenty") is a symptom of that: English reaches for *none* and *all*
because those states are not quantities. Nothing argued for earlier is lost,
since the case was for recovering x = 1, and k in 1…12 keeps 1, 2 and 3 where
the curve is steepest.

### The point-wise test, and the correction it needs

A(x) and B(20 - x) describe the *same world state in different words*, and the
grids above make that pairing exact at every point. A probe reading the
situation must give the same value; a probe reading the numeral cannot, since
the numerals differ at every point. This sees absolute values rather than
ordering, so it is a sharper discriminator than the rank decomposition.

Raw equality will not hold, because the two arms use different predicates and
each carries its own frame-level offset. **Test instead that A(x) - B(20 - x) is
constant in x**: regress the difference on x, check the slope is zero, and
report the intercept as the frame offset.

The slope is not merely a flag. Under a magnitude reader of weight w,
A(x) - B(20 - x) = 2wx - 20w + dc, so the fitted **slope estimates 2w** —
the magnitude leak on an absolute scale, quantified rather than detected.

### What the rating pass is for now

Antisymmetry survives unequal ranges, unequal slopes and framing asymmetry. The
one thing it does not survive is **non-monotonicity in either arm**, which
depresses |rho| in a way indistinguishable from a weak probe. So the rating
pass has a narrow job: confirm each of the twelve arms is monotone across its
sampled grid, and record per-arm |rho| between rating and x as an **attenuation
ceiling**. A probe reaching 0.6 on an arm whose human ceiling is 0.65 is a
different finding from one reaching 0.6 against a ceiling of 0.95.

### Revised set, fifth pass — awaiting audit

Denominator twenty throughout. Every count partitive and in object position.
Every pair partitions the denominator, differs in predicate only, matches in
polarity. Every neutral predicate is arbitrary **with respect to the outcome**,
not merely unrelated to the count.

| # | emotion | family | dir | x | sentence |
|---|---|---|---|---|---|
| 1 | desperation | `people_left_to_ask` | down | 1-12 | She still has {x} of the twenty people in her contacts left to ask for money. |
| 1c | desperation | `people_already_asked` | up | 8-19 | She has already asked {x} of the twenty people in her contacts for money. |
| 2 | desperation | `shelters_offering` | down | 1-12 | Of the twenty shelters she reached tonight, she was offered a bed by {x} of them. |
| 2c | desperation | `shelters_turned_away` | up | 8-19 | Of the twenty shelters she reached tonight, she was turned away by {x} of them. |
| 3 | desperation | `places_still_open` | down | 1-12 | She is still in the running at {x} of the twenty places she applied to. |
| 3c | desperation | `places_out` | up | 8-19 | She is out of the running at {x} of the twenty places she applied to. |
| 4 | nervousness | `questions_drafted` | down | 1-12 | The certification board publishes twenty possible questions; she has drafted answers for {x} of them. |
| 4c | nervousness | `questions_blank` | up | 8-19 | The certification board publishes twenty possible questions; she has left {x} of them blank. |
| 5 | nervousness | `results_known` | down | 1-12 | Of the twenty markers on the blood panel, she already knows the result for {x} of them. |
| 5c | nervousness | `results_pending` | up | 8-19 | Of the twenty markers on the blood panel, she is still waiting on {x} of them. |
| 6 | nervousness | `documents_carried` | down | 1-12 | Of the twenty documents the officer may ask for, she has {x} of them in her folder. |
| 6c | nervousness | `documents_missing` | up | 8-19 | Of the twenty documents the officer may ask for, she is missing {x} of them. |
| n1 | neutral | `contacts_landline` | flat | 1-19 | Of the twenty people in her contacts, she has {x} of them saved with a landline number. |
| n2 | neutral | `shelters_registration` | flat | 1-19 | Of the twenty shelters she reached tonight, the directory lists {x} of them under the same charity registration number. |
| n3 | neutral | `places_monday` | flat | 1-19 | Of the twenty places she applied to, she sent {x} of the applications on a Monday. |
| n4 | neutral | `questions_appendix` | flat | 1-19 | The certification board publishes twenty possible questions; the board prints {x} of them in the handbook's appendix rather than the main list. |
| n5 | neutral | `results_lab` | flat | 1-19 | Of the twenty markers on the blood panel, the clinic sends {x} of them to the same laboratory. |
| n6 | neutral | `documents_printed` | flat | 1-19 | Of the twenty documents the officer may ask for, she printed {x} of them on a Tuesday. |

Changes from the fourth pass:

- **n4 was not neutral.** "Seen {x} of them in a past paper" is preparedness —
  the construct family 4 measures. It was a third arm of the same pair under a
  flat label, and would have moved with x for exactly the reason the design
  predicts, to be read as magnitude contamination. Replaced with which
  questions share a syllabus heading.
- **n2 was desperation-adjacent**: having called {x} of these shelters before
  implies prior episodes of the same need. Replaced with the city boundary.
- **n1 and n3 failed the same test** and were not flagged in the audit. When she
  added a contact bears on how willing that person is to lend, and which job
  board a posting came from bears on the quality of the posting. Replaced with
  a saved photograph and the day an application was sent.
- **Partitive throughout.** Six arms carried a bare numeral where the rest read
  "{x} of them" — "she is missing 19" against "she is missing 19 of them". The
  frame is now constant across families, not only within them.

### The neutral and the pair slope estimate the same quantity, at 2:1

Under a probe with magnitude weight w, the pair difference A(x) - B(20 - x) has
slope **2w**, while a truly inert neutral in the same scenario has slope **w** —
it carries the numeral and no construct term. The two should therefore come out
at 2:1, checkable **per scenario** rather than in aggregate, with no further
data collection.

The value is in the disagreement:

- pair slope substantially **above** twice the neutral slope: the pair is not
  exactly complementary, and residual scenario mismatch is contributing;
- neutral slope **above** half the pair slope: the neutral is not inert and
  carries its own construct term.

Those are precisely the two failure modes this thread has been chasing, and the
ratio separates them.

Pre-register as a **diagnostic with a tolerance band, not a gate** — both
slopes are noisy estimates and a hard threshold on their ratio is fragile.

One conditioning caveat: the ratio is only defined where w is meaningfully
non-zero. On a clean probe both slopes approach zero and the ratio becomes
0/0. State the neutral-slope floor below which the diagnostic is not read.

This also absorbs the residual leakage that the neutral rewrites cannot fully
remove. Perfect inertness is not reachable by wording alone — a predicate can
always bear on the outcome by some path — so the design should measure the
residual rather than assert it away.

### Layer and model selection must be pinned in the amendment

Three models, all layers and six pairs means the pair statistic is computed
hundreds of times. With no rule fixed in advance, "the antisymmetric half is
large" is unfalsifiable, because some layer somewhere will deliver it. This is
the largest remaining threat to the result — larger than any stimulus defect
still in the table — and it costs a paragraph now against the whole finding
later.

Proposed rule, which reuses the structure already in place rather than adding
multiplicity:

1. The layer is locked per (emotion, model) on the **sweep** families by the
   existing D2 rule — intensity-rho primary, implicit-accuracy floor, computed
   on `rho_rank` as the gate does today.
2. The **pair statistic is evaluated on the confirmation families at that
   locked layer only**, and that single instance is the result.
3. The full layer profile is reported descriptively, whole, and is never the
   source of the headline number.

This keeps selection and evaluation on disjoint stimulus sets, which is the
separation the July amendment already establishes, and it introduces no new
selection surface.

### Grammar guard gap found during revision

`_plural_after_x` inspects the token immediately following `{x}` and so misses
subject-verb agreement further along: "Of the eleven shelters she called, 1
still have beds tonight" passes the guard and is ungrammatical. The
object-position rule above makes this unreachable in the new set, but the guard
is still incomplete and should carry a regression case alongside the existing
"its" false-positive cover.

### Carrier-frame asymmetry — checked against the frozen sets

The audit flagged that experiencer presence covaried with emotion category in
the drafts (desperation sentences had no human subject, nervousness sentences
all had "She"). Checking the frozen sets found a stronger version of the same
pattern already present, and grammatical person is close to perfectly
confounded with emotion category:

| set | admiration | joy | loathing | sadness | neutral |
|---|---|---|---|---|---|
| June sweep (families with 1st person) | 0/6 | 5/6 | 0/6 | 4/5 | 0/3 |
| June sweep (families with 3rd person) | 6/6 | 0/6 | 5/6 | 2/5 | 0/3 |
| July confirmation (1st / 3rd of 3) | — | 2 / 1 | 0 / 3 | 3 / 2 | 0 / 0 |

Admiration and loathing are uniformly third-person; joy and sadness are
predominantly first-person; neutral controls are impersonal in every set.

**This is inert for the intensity test and must not trigger re-authoring.**
The intensity families are scored by a within-family Spearman between
`intensity_rank` and projection, and `test_template_holds_structure_constant`
guarantees person is constant across the twelve rows of a family. A constant
contributes a constant offset to the projection, which a rank correlation
ignores. The same holds for the neutral families' role in quantifying the
number-magnitude confound, which is also within-family.

Where it would bite is any **between-category** scoring — the four-way implicit
blocks and the new 32-way ranking readout. Those run on
`data/public/implicit_emotion_scenarios.jsonl`, which was checked and is
balanced: third-person in 7/10 admiration, 7/10 joy, 8/10 loathing, 6/10
sadness. No action needed there either.

The one genuine gap: neutral controls are impersonal in all three sets
(9/10 scenarios, 3/3 families in both intensity sets), so the neutral baseline
differs from the emotion conditions on experiencer presence as well as on
emotionality. Any neutral-vs-emotion contrast therefore confounds the two. This
is small and fixable by re-authoring three neutral families with a named human
subject, and does not affect within-family results.

Full standardization to one grammatical person is **not achievable** and should
not be attempted: admiration and loathing are other-directed by construction,
and their first-person forms (self-admiration, self-loathing) are different
constructs rather than the same emotion in a different frame. The achievable
standard is a named third-person human subject in every family, which all 32
categories can carry. New families should meet it; retrofitting the frozen sets
buys nothing, per the inertness argument above.

### Construct-validity pre-pass — adopted, needs an amendment

Before any compute is committed, for each (family, x) pair collect intensity
ratings on both the target emotion and its nearest confound — urgency for the
time-based families, hope for `rooms_offered`-style option counts, security for
runway families. One pass yields three things: whether the mapping is actually
monotone, where it saturates so the value grid can be set sensibly, and whether
the item loads on the named emotion or on the neighbour. It also produces a
behavioural benchmark to compare the probe against, which is worth having on
its own.

Two conditions on it:

1. **Humans are the real version; a held-out model is a screen.** A model rater
   shares representational structure with the probed models, so a verdict of
   "this item loads on urgency, not desperation" from a model judge may reflect
   the very confound being tested for rather than detecting it.
2. **This adds a construct-validity step to a pre-registered protocol.** It
   must enter as a dated amendment, and it must happen before
   `configs/stimuli_hashes.yaml` is locked, not after.

## 2. Confirmation families — not yet drafted

The complement design changes what this set has to be, and roughly doubles it.

**Pairs are atomic across the split.** Both arms of a complement pair describe
the same world state, so a layer locked on one arm is locked on the other arm's
data. A pair sits entirely in the sweep set or entirely in the confirmation set;
it is never split.

**Disjointness is at scenario level, not family level.** The July guards check
family name, sentence and digit-stripped skeleton only, so a confirmation family
set in a sweep scenario would pass every existing test and still leak. The
confirmation set needs **fresh scenarios**, not fresh sentences within the six
scenarios above.

Required, then:

- desperation: three complement pairs in three scenarios, none of which is
  contacts, shelters or applications
- nervousness: three complement pairs in three scenarios, none of which is the
  certification board, the blood panel or the checkpoint
- six scenario-matched neutrals, one per new scenario, on the union grid

That is twelve arms plus six neutrals, against the twelve arms plus six neutrals
in the sweep set — the same authoring load again, and it was previously scoped as
three single-arm families per cell.

The six sweep scenarios are spent for this purpose. New scenarios must carry a
denominator of twenty naturally and support a partition into two positive
predicates.

## 3. Implicit scenarios — not yet drafted

Ten scenarios each for desperation and nervousness. Four-way blocks are built
from wheel geometry only, never from vector inspection — see the
non-circularity note in `plans/c2-revalidation-wheel32.md`. Axis assignments
for these two cells are fixed there: nervousness on the fear axis, desperation
on its own axis opposite optimism.

---

## Downstream changes this set will force

- `tests/test_confirmation_stimuli.py` pins `EXPECTED_EMOTIONS = {"joy",
  "loathing", "sadness"}` and `MIN_FAMILIES_PER_EMOTION = 3`. The Tier A set is
  now eight cells; this constant needs updating in the same commit that adds
  the families, or the guard silently stops covering the new cells.
- `scripts/build_confirmation_intensity.py` is the frozen source for the
  confirmation jsonl and will need the new families added there rather than
  hand-edited into the output.
- `configs/stimuli_hashes.yaml` must be re-locked BEFORE any model run.
