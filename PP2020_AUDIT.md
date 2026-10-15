# Plant Pathology 2020 field-training audit

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\audit_pp2020_for_field_training.py
```

The default output is `data/field_manifests/pp2020_audit/`. It must be new or
empty. To repeat without overwriting an earlier audit:

```powershell
.venv\Scripts\python.exe scripts\audit_pp2020_for_field_training.py --output-dir data\field_manifests\pp2020_audit_rerun
```

The script reads the existing local `extracted/train.csv` and images. It never
uses challenge test labels, copies images, writes resized sources, assigns
splits, or trains a model. Source/registry byte hashes are checked again before
the metadata outputs are written. Missing required manifests, unexpected
previous-use counts, unreadable registry images and nonempty output folders
stop the run. Missing/inconsistent broader evidence blocks candidate eligibility.

## Outputs

- `pp2020_inventory.csv`: every labelled record, original flags, independent
  label/ID/image statuses, original mode and dimensions, EXIF orientation,
  RGB-conversion result, SHA-256, oriented pixel SHA-256 and 256-bit dHash.
- `previous_use_registry.csv`: the 40 healthy segmentation training images,
  50 evaluation-only healthy images and existing cedar extraction files, with
  provenance and content hashes.
- `duplicate_screening.csv`: ID/content/near-duplicate links to previous use,
  plus within-dataset duplicate links. Near-duplicate threshold is Hamming
  distance 4/256 by default; hits require review rather than automatic relabelling.
- `training_candidates.csv`: valid single-label records without blocking issues.
- `quarantine_and_exclusions.csv`: all excluded/held rows and every reason.
- `audit_summary.json`: counts, evidence references, unresolved history and checks.

Perceptual hashing uses an EXIF-oriented, in-memory RGB copy, converted to
grayscale and resized to 17x16 for dHash. Original dimensions/mode and raw-file
SHA-256 remain separately recorded. No source file is rewritten.

Renamed and metadata-only copies are screened by content, not just filenames.
Evaluation reservations and previous-use roles propagate across duplicate groups.
The 40 previously trained healthy images can be training candidates but cannot
be treated as unseen evaluation examples. No splits are assigned here.

## Cedar prior-use limitation

The existing extraction contains 622 rust images. The audit scans project
scripts, documentation and saved metadata for `cedar_pp2020` references. A
reference proves neither evaluation nor model selection. In this run only the
extraction script was found. Its use history remains unresolved: all clean rust
rows remain held, even if a particular extracted copy is absent. Their registry
role is `cedar_prior_use_unresolved`; do not infer that they are fresh training
or independent evaluation data. A later documented history review is required.

## Local results, 2026-10-16

- 1,821 labelled records; all images decoded and were RGB-compatible.
- 40 previous training records and 50 reserved evaluation records identified.
- 622 rust records held pending cedar prior-use review.
- 91 multiple-disease records excluded from C1.
- One exact duplicate pair: `Train_379` (scab) and `Train_1173`
  (multiple diseases). Both are held; labels are not guessed or changed.
- 1,057 provisional training candidates: 591 scab and 466 healthy.
- No registry-integrity/evidence-scan errors; source and registry file hashes
  were rechecked and unchanged. No images downloaded, copied or modified.

These candidates are not a complete four-class training set. Rust history must
be resolved, and PP2020 supplies no black-rot class here. This audit is not a
training run or split assignment.

All 24 repository tests passed, including audit fixtures for corrupt images,
label errors, renamed content duplicates, near-duplicate evaluation reservations,
missing manifests, count mismatches, raw/EXIF dimensions and source preservation.
