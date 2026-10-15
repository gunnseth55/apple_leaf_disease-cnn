# Task B: PP2020 duplicate adjudication

Output: `data/field_manifests/pp2020_duplicate_adjudication/`.

## Results

All **713 relations were confirmed by recomputed exact pixel SHA-256**, with
complete PIL decoding and source-byte identity checked against the original
audit. All recomputed 256-bit dHash distances are zero. There are **zero dHash-only
candidates and zero false-positive pairs** in these inputs. A dHash near match
alone would not be automatically accepted as a duplicate.

- 712 relations are original-to-copy aliases: 622 Cedar copies, 40 previous
  segmentation-training copies and 50 evaluation-only healthy copies.
- One relation joins two different labelled records: `Train_379` (scab) and
  `Train_1173` (multiple diseases). Their bytes and decoded pixels match, but
  their original one-hot label signatures disagree. Both remain quarantined;
  no diagnosis is guessed and neither may receive a split until resolved.
- The graph contains 713 duplicate components and 1,107 singleton components,
  for 1,820 total components across 2,533 file nodes. Only one component contains
  more than one labelled PP2020 record. Copy aliases are not independent samples.

## Visual evidence

`contact_sheet_index.html` links all 60 contact-sheet pages, covering every pair
exactly once. Each tile shows both images, image IDs, source labels, recorded
roles, recomputed distance, byte/pixel equality and the decision. Conflicting
labels appear first. The conflict has a separate full tile,
`PAIR_0713_label_conflict.png`.

Thumbnails are rendered previews, never training inputs. Exact-identity decisions
are based on complete decoded pixels, not thumbnail similarity. Layout and the
conflicting pair were visually inspected; no claim is made that 713 separate
manual biological diagnoses were performed. No dHash-only visual decisions were
needed. Contact-sheet roles retain original-audit provenance; the historical
`cedar_prior_use_unresolved` role is superseded for permissions by the verified
Task A clearance, recorded as `cedar_task_a_clearance_used=True`.

## Outputs and split requirements

- `pp2020_duplicate_adjudications.csv`: 713 pair decisions and contact-sheet locations.
- `exact_pixel_matches.csv`: all confirmed exact-pixel pairs.
- `dhash_candidates.csv`, `false_positive_pairs.csv`: empty, with headers.
- `pp2020_duplicate_group_members.csv`: file-level membership and original IDs.
- `pp2020_split_group_constraints.csv`: one row per connected component.
- `quarantined_group_members.csv`: the two conflicting source records.
- `adjudication_summary.json`: counts and input metadata hashes.

Split **group IDs**, never individual rows or copied paths. Select one canonical
source file per image; do not add extracted aliases as additional samples.
Every confirmed group must remain within one split. The group constraints are:

| Status | Components | Allowed allocation |
| --- | ---: | --- |
| Accepted | 1,639 | One prospective train, validation or test split |
| Previously trained | 40 | Training only for an independent final-pipeline evaluation |
| Reserved evaluation | 50 | Evaluation only; never training or validation |
| Multiple diseases excluded from C1 | 90 | No C1 split |
| Conflicting labels | 1 | Quarantine both records until adjudicated |

The 90 excluded singleton groups plus the multiple-disease member in the
conflicting group account for all 91 multiple-disease source records. Task A
clearance is conditioned on available local use-history evidence. This step
does not create any splits or change existing manifests.

`enforce_split_constraints()` rejects mixed-split groups, unknown groups,
quarantined assignments, evaluation-to-training leakage and reuse of the 40
previously trained groups as independent evaluation. Future split-generation
code must use this validator and the group table.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\adjudicate_pp2020_duplicates.py --output-dir data\field_manifests\pp2020_duplicate_adjudication_rerun
```

Use a new/empty output directory. For any future nonexact dHash candidates,
populate `visual_decisions_template.csv` with a decision, reason and reviewer,
and pass it with `--visual-decisions`. Unreviewed candidates remain held and
are not joined into confirmed components. All 32 repository tests passed,
including identity-vs-dHash decisions, transitive grouping, conflicting labels,
evaluation reservations and split-constraint enforcement.
