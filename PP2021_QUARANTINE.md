# Task C2-B: PP2021 internal-conflict quarantine

The metadata-only quarantine is complete. No training, inference, diagnostic
relabeling, source-image edit, checkpoint edit, or split assignment was performed.

## Verified result

| Source label | Original candidates | Quarantined candidates | Remaining pending candidates |
| --- | ---: | ---: | ---: |
| `scab` | 4,826 | 0 | 4,826 |
| `frog_eye_leaf_spot` | 3,181 | 0 | 3,181 |
| `rust` | 1,860 | 27 | 1,833 |
| `healthy` | 4,624 | 0 | 4,624 |
| **Total** | **14,491** | **27** | **14,464** |

All **27** internal exact-pixel duplicate groups have conflicting original
labels. All **54** original records in those groups are quarantined, including
the 27 Rust candidates and 27 excluded-label records. The latter retain their
original labels: 20 `complex`, six `rust complex`, and one `rust frog_eye_leaf_spot`.
No choice was made about which label is correct.

The 14,464 remaining records are **not approved for training**. Their status is
`pending_cross_dataset_audit`, and both `eligible_for_training` and
`eligible_for_unseen_evaluation` are false. Frogeye-to-black-rot remains a
provisional diagnostic mapping, explicitly identified in `mapping_status`.

## Implementation and actual input schemas

Script: [scripts/quarantine_pp2021_conflicts.py](scripts/quarantine_pp2021_conflicts.py).

The script reads the five actual input files as strings without replacing empty
fields with NaN. The original `image` filename is the record identifier; it is
preserved with its extension.

| Input | Relevant schema |
| --- | --- |
| `data/plant_pathology_2021/train.csv` | `image, labels` |
| `data/plant_pathology_2021/pp2021_single_label_candidates.csv` | `image, labels, proposed_label, review_status` |
| `data/field_manifests/pp2021_audit/image_inventory.csv` | `image, original_labels, proposed_label, status, width, height, file_sha256, pixel_sha256, dhash256, error` |
| `data/field_manifests/pp2021_audit/internal_duplicate_groups.csv` | `group_id, image_count, image_ids, original_labels, label_conflict, candidate_count, excluded_count, status` |
| `data/field_manifests/pp2021_audit/internal_duplicate_records.csv` | all inventory fields, plus `group_id, label_conflict` |

The transformation checks 18,632 unique original IDs, complete inventory
coverage, zero saved image errors, 14,491 source-approved single-label
candidates, original/proposed label agreement, and 18,605 unique saved pixel
hashes. It independently derives repeated-pixel components from the saved
inventory and reconciles every component with both duplicate reports. Each
group must have two exact-pixel records, two original label strings, one
candidate, and one excluded record. Strict count, schema, membership, and
boolean checks fail before publication when they disagree.

The complete 54-record quarantine is an **audit overlay**: it supersedes the
earlier `ok`/`excluded_label` status for those records without rewriting the
original inventory. Records carry original labels, proposed label, original
audit/review status, hashes, original group ID, candidate membership, and the
reason code `exact_pixel_duplicate_conflicting_original_labels`. Both `status`
and `review_status` are `quarantined_label_conflict` for all 54 records.

Candidate row order is preserved. Quarantined originals are ordered by group
and filename. Excluded records retain empty proposed labels. Remaining candidate
groups are blank because all reported internal duplicate pairs were quarantined;
this does not certify independence from external datasets or near-duplicates.

## Generated files and version safety

Two older partial outputs already existed at the audit root. They contain only
the four original candidate columns and lack group/hash/reason provenance.
They were preserved byte-for-byte. The complete new audit therefore uses an
explicit version under `data/field_manifests/pp2021_audit/internal_quarantine_v1/`:

- [quarantined_conflicting_records.csv](data/field_manifests/pp2021_audit/internal_quarantine_v1/quarantined_conflicting_records.csv): all 54 original records.
- [quarantined_conflicting_candidates.csv](data/field_manifests/pp2021_audit/internal_quarantine_v1/quarantined_conflicting_candidates.csv): the 27 removed candidates.
- [candidates_after_internal_audit.csv](data/field_manifests/pp2021_audit/internal_quarantine_v1/candidates_after_internal_audit.csv): 14,464 pending candidates.
- [internal_quarantine_report.json](data/field_manifests/pp2021_audit/internal_quarantine_v1/internal_quarantine_report.json): counts, all 27 group IDs, reasons, source/script/output hashes, audit state, and preservation evidence.

The original 14,491-candidate manifest is unchanged. Output CSVs use deterministic
UTF-8/LF serialization. New versions are staged and published as a complete
directory. Existing files are never silently overwritten: an identical audit
is verified and reused; a different/incomplete audit requires an explicitly new
`internal_quarantine_vN` output directory. The actual v1 command was rerun and
reported `verified_and_reused_identical_audit` without rewriting the outputs.

```powershell
.venv\Scripts\python.exe scripts/quarantine_pp2021_conflicts.py --output-dir data/field_manifests/pp2021_audit/internal_quarantine_v1
```

The same version is also the script's default. Future changed source inputs,
implementation, or recorded preservation state may require a new version;
never overwrite v1 to accommodate them.

## Tests and preservation evidence

[tests/test_pp2021_quarantine.py](tests/test_pp2021_quarantine.py) adds nine tests.
They verify the real counts (27/54/27/14,464), disjoint output membership, unique
IDs, unchanged source labels, preserved provisional mapping, group/reason
provenance, class reconciliation, and pending permissions. Failure tests cover
tampered labels, hashes, group summaries, booleans, missing members/images,
duplicate IDs, unsafe output paths, and changed protected checkpoints. A fixture
compares original file bytes before/after publication, tests no-write reuse,
refusal of changed existing outputs, and safe publication of a new version.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_pp2021_quarantine.py
.venv\Scripts\python.exe -m unittest discover -s tests
```

Results: **9/9 new tests passed; 63/63 total repository tests passed**.

The actual run compared before/after SHA-256 for **417 metadata/checkpoint files**
across source manifests, previous audits, frozen splits, artifacts, results, and
lesion manifests. No changes were found. It also checked path, size, mtime and
ctime for **29,446 image/archive paths**, including the external cached PV
source images; no changes were found. These counts and fingerprints are in the
JSON report. All input manifests and legacy partial outputs retain their hashes.

No PP2021 image was decoded or content-hashed again. Image preservation is
supported by read-only transformation, file-metadata comparisons and byte-level
fixture tests; it is **not a new byte-level verification of all dataset images**.
Current image-content identities are taken from the supplied verified inventory.
The initial audit script and its outputs were not rerun or modified.

## Next task

Update, 2026-10-23: C2-C matching is complete. See
[PP2021_CROSS_DATASET_AUDIT.md](PP2021_CROSS_DATASET_AUDIT.md). It found no
cross-dataset raw/pixel exact matches and 1,293 perceptual pairs needing review;
595 of the 14,464 remaining candidates have visual holds. The C2-B outputs
documented above remain unchanged. The following paragraph records the earlier
handoff plan.

See [PP2021_CROSS_DATASET_PLAN.md](PP2021_CROSS_DATASET_PLAN.md) for C2-C. The next
implementation step is to create `scripts/audit_pp2021_cross_dataset.py` with
source-role adapters, explicit hash-method compatibility, indexed exact/dHash
matching, reviewed-link group restrictions, and fail-closed coverage tests.
That script has **not** been implemented or run in C2-B. No C2 splits exist.
