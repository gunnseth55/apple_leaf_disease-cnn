# PP2020 Cedar usage review

The review output is
`data/field_manifests/pp2020_cedar_usage_review_verified/pp2020_cedar_usage_review.csv`.
It contains one row per clean rust-labelled PP2020 image, including the image ID,
prior role, evidence files, status, allowed future uses and verified source/copy
identity. Supporting CSVs enumerate every scanned metadata file and evidence
match. A JSON snapshot preserves the reviewed checkpoint metadata.

## Interpretation

Copying an image does not constitute training, evaluation or model selection.
`prepare_cedar_external.py` filters the label CSV and uses `shutil.copy2`; it has
no model-loading, inference, optimization or selection code. Source/copy SHA-256
equality verifies the extraction for each of the 622 files.

`copied` with status
`copy_verified_no_other_use_found_in_reviewed_records` means copying is confirmed
and no additional use was found in the recorded local history. It does not prove
that unrecorded, deleted or external uses never occurred. Extraction alone is
not grounds for excluding these images from all future supervised candidates.

Allowed prospective uses are training, validation **or** a future test split,
subject to the final protocol and duplicate-group separation. Assign each image
one role. Lock any new test split before further development and do not tune
against it. If additional prior-use evidence emerges, revise these permissions.

Recorded model-selection/training use prohibits claiming the image is a fresh
independent test example for that development process. Recorded evaluation is
also kept distinct from fresh testing. A manifest alone is not treated as proof
that inference or optimization actually ran: unresolved memberships are held
for contextual review.

## Evidence scope

- Original labels, the earlier audit inventory, and all 622 extracted copies.
- Current local CSV manifests, predictions, per-image metrics and evaluation
  records, plus JSON metadata, notebooks, scripts, logs and documentation.
- Image IDs, raw SHA-256 and EXIF-oriented decoded-pixel SHA-256 matches, including
  renamed/metadata-only copies among available image records.
- All local `artifacts/**/*.pt` checkpoint metadata, without constructing or
  running models. The recorded EfficientNet selection uses PlantVillage
  validation macro-F1; the lesion workflow uses validation Dice and threshold
  calibration on lesion validation data.
- Git history of Cedar files and historical metadata changes mentioning
  `cedar_pp2020` or PP2020 `Train_...` identities.
- Relevant persisted PowerShell history references; history is not a complete
  execution log and commands alone do not prove successful completion.

The original audit and its conservative Cedar hold are preserved as historical
outputs. The usage-review CSV is the later evidence layer; the script does not
modify the old candidate lists, create new splits or train/evaluate any model.
The 50 healthy evaluation-only reservations remain unaffected.

## Result, 2026-10-16

All 622 images have status
`copy_verified_no_other_use_found_in_reviewed_records` and prior role `copied`.
No matching Cedar training, prediction/evaluation or model-selection record was
found in the reviewed evidence. All source/copy hashes matched the earlier audit.
The completed review checked 347 local metadata files, 5,192 referenced image
identities, 14 historical metadata blobs and 13 checkpoint records, with zero
coverage errors. Git shows the extraction script added in commit
`fba91889f9882294fca1a7fcc2eeb5f8357d086e`; that commit's Kashmir evaluation
title does not establish Cedar evaluation.

These 622 records are therefore eligible for prospective split allocation under
the stated local-evidence scope, duplicate separation and final protocol.
The earlier blanket hold based on extraction alone is not evidence of
contamination. No old audit outputs were overwritten or split memberships created.
All 28 repository tests passed, including four usage-review tests.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\review_pp2020_cedar_usage.py --output-dir data\field_manifests\cedar_usage_review_rerun
```

Choose a new/empty output folder. Unexpected labels, source/copy hash mismatches
and incomplete evidence coverage prevent a copy-only clearance. Read
`review_summary.json` for counts and coverage errors. No image files are copied
or modified by the review.
