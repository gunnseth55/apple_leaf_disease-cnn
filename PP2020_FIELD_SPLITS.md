# Task D: frozen PP2020 field splits

The frozen version is `data/field_manifests/pp2020_split_v1/`. No model training
or epochs were run to create it.

## Commands

From the repository root, verify the already-created version before training:

```powershell
.venv\Scripts\python.exe scripts\build_pp2020_field_splits.py --verify
```

On a checkout where v1 has not yet been built:

```powershell
.venv\Scripts\python.exe scripts\build_pp2020_field_splits.py --seed 42
```

The builder refuses to overwrite an existing version, even with a different seed.
Verification checks current audit inputs, all manifest/report hashes, role and
group restrictions, source file hashes, and deterministic replay. It does not
rewrite the frozen files. A deliberate experiment change must use a new version,
for example `--output-dir data\field_manifests\pp2020_split_v2`. Do not change
split versions in response to test performance.

You can use the virtual environment Python directly; activation is unnecessary.
For Command Prompt, `call .venv\Scripts\activate.bat` also works. For PowerShell,
use `.venv\Scripts\Activate.ps1` instead of the batch activation script.

## Frozen counts

| Class | Model index | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple scab | 0 | 414 | 88 | 89 | 591 |
| Cedar apple rust | 2 | 436 | 93 | 93 | 622 |
| Healthy | 3 | 326 | 70 | 70 | 466 |
| **Total** | | **1,176** | **251** | **252** | **1,679** |

The full four-class mapping is preserved: black rot remains model index 1, with
zero PP2020 examples. This is not a claim of field black-rot supervision.

## Restrictions and accounting

- The 50 earlier evaluation-only healthy groups and their 50 confirmed copies
  remain outside **all** new splits. They stay an existing benchmark, not a newly
  unseen test set.
- All 40 healthy images seen during segmentation training are assigned to
  classification training. They are not unseen combined-system validation/test.
- `Train_379` and `Train_1173` remain quarantined for conflicting labels.
- Each accepted confirmed duplicate group supplies one canonical original
  photograph. Copied paths never inflate training/evaluation counts.
- `excluded.csv` has 854 file records: 142 original records plus 712 extracted
  copy aliases. Thus 1,679 accepted originals + 142 excluded originals = all
  1,821 audited labels, and 1,679 + 854 = all 2,533 file members.
- Excluded copy aliases record `inherited_group_split` when their canonical
  source was assigned. This records identity propagation, not actual inclusion
  of the copy in a training or evaluation manifest.

## Allocation and freeze procedure

The builder joins the original classification role registry to exploded
duplicate-group constraints by original image ID and joins the 622 Cedar usage
reviews by the same ID. It validates original labels, Task A/C permissions,
Task B/C provenance hashes, copy memberships and live prior-use manifests.
Missing or contradictory permissions stop the build before publication.

The custom group allocator computes integer 70/15/15 targets for each complete
eligible class with the Hamilton largest-remainder rule. Equal fractional ties
prefer train, test, then validation. Mandatory training rows are charged against
their class's training quota. Remaining groups are seeded-shuffled (seed 42),
processed by descending size with seeded tie order, and assigned by the smallest
incremental squared quota error. Confirmed groups never split. This preserves
the overall class proportions while accommodating the 40 forced training rows.
It does not run StratifiedGroupKFold or optimize any model-performance metric.

The minor difference from the preliminary planning table follows deterministic
integer rounding: Cedar has 436/93/93 rather than 435/93/94. Seed, algorithm ID,
implementation hash, exact source CSV/JSON hashes, class mapping, counts and all
four manifest hashes are saved in `split_report.json`. Integrity results and
the report hash are saved in `split_integrity_report.json`. All six requested
files are published together only after validation.

## Outputs

```text
pp2020_split_v1/
  train.csv
  val.csv
  test.csv
  excluded.csv
  split_report.json
  split_integrity_report.json
```

Accepted manifests contain original IDs, paths, group IDs, class names/indices,
source and pixel hashes, previous segmenter exposure and split. `path` and
`label` are compatible with the existing classification dataset interface.

All 42 repository tests passed. Six new tests cover integer targets, seeded
replay independent of row order, mandatory training-only groups, indivisible
groups, canonicalization, label conflicts, missing boolean permissions and
overwrite refusal. The actual v1 freeze and a separate `--verify` run both passed.

## Next experiment

The mixed-source fine-tuning stage is separate work. It should initialize a new
run from the frozen reference EfficientNet-B0, preserve its original checkpoint,
use only PlantVillage train and this PP2020 train manifest for optimization,
and validate each source separately. Neither PP2020 test nor PlantVillage test
may guide checkpoint selection or tuning. PP2020 supplies no black-rot field
category. No mixed-source training command is introduced by Task D.

The subsequent user-run C1 experiment completed on 2026-10-18 using these frozen
manifests. Epoch 10 was selected with PV validation macro-F1 1.0000, PP2020
present-three-class validation macro-F1 0.9508 and combined score 0.9754. The
manifest/report hashes still match the training configuration; PP2020 test remains
unevaluated for C1. See [FIELD_CLASSIFIER_TRAINING.md](FIELD_CLASSIFIER_TRAINING.md)
for complete results and checkpoint provenance. No split was changed.
