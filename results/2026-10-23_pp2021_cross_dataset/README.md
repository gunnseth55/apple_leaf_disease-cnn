# C2-C cross-dataset matching snapshot

Full report: [PP2021_CROSS_DATASET_AUDIT.md](../../PP2021_CROSS_DATASET_AUDIT.md).
Full local audit: `data/field_manifests/pp2021_cross_dataset_v1/`.

Matched all 18,632 PP2021 records against 6,958 reference photograph paths.
No cross-dataset raw-file or normalized-pixel exact duplicates were found.
The 27 original internal pairs remain quarantined. The dHash256 radius-four
screen found 1,293 nonexact PP2021/PP2020 pairs involving 706 PP2021 records;
all decisions remain unresolved.

Of the 14,464 C2-B remaining candidates, 595 have visual-review holds, including
176 with potential protected validation/test links. Another 3,181 require
provisional Frogeye-to-black-rot diagnostic mapping review. The remaining 10,688
have no detected overlap but are not approved for training.

- `priority_review_queue.csv` links every pending pair to its sheet/tile and
  inherits reference roles through confirmed groups, including copied aliases.
  Priority 1 covers possible protected-history matches, priority 2 prior-training
  matches, priority 3 other unresolved matches. Priority is not adjudication.
- `review_priority_summary.json` contains queue counters and screening distances.
- `verification_report.json` records independent graph and alternate-index checks.
- The three copied audit JSON reports preserve counts, source/output hashes,
  hash methods and preservation evidence.
- `snapshot_sha256.json` hashes the generated data files, excluding itself and
  this explanatory README.

Sheet paths in the queue resolve relative to the full local audit directory.
No source images or checkpoint weights are included in this compact snapshot.
Recreate or verify identical generated snapshot files with:

```powershell
.venv\Scripts\python.exe scripts/summarize_pp2021_cross_dataset.py
```

The script first verifies the complete local audit, refuses changed existing
snapshot data, and writes no model/dataset/split changes. All 79 project tests
and the independent completed-audit verifier passed. Next: documented visual
adjudication, prioritizing protected-history links; no C2 splits or training yet.
