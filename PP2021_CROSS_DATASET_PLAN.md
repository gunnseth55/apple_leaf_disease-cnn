# Task C2-C implementation plan: PP2021 cross-dataset overlap verification

Update, 2026-10-23: matching has been implemented and completed. See
[PP2021_CROSS_DATASET_AUDIT.md](PP2021_CROSS_DATASET_AUDIT.md) for the actual
coverage, method, verified results and review handoff. The text below preserves
the preceding implementation plan; visual adjudication remains pending.

Status: plan only, following the passing C2-B quarantine. Target implementation:
`scripts/audit_pp2021_cross_dataset.py`. This task has not performed cross-dataset
matching, assigned C2 splits, or trained any model.

## Scope and invariant

Query **all 18,632 PP2021 inventory records**, including excluded labels and all
54 quarantined records. The 14,464 pending candidates are an eligibility view,
not the matching universe. Excluded or quarantined nodes may connect candidates
to protected data through exact or reviewed duplicate relationships.

Start by verifying the C2-B source/output hashes, group membership and statuses.
Carry original source labels, proposed mapping and mapping uncertainty separately;
never erase a source-label conflict by first collapsing labels to four outputs.
Unconfirmed Frogeye/black-rot equivalence cannot resolve a diagnostic conflict.

## Source registry adapters and previous-use evidence

Build a registry with dataset-qualified node IDs, resolved paths, original IDs,
raw labels, proposed labels, source file/row evidence, all known roles, exposure
to classifier/segmenter training and selection, historical evaluation exposure,
group IDs, hash values and explicit hash-method identifiers. Keep a separate
node-to-evidence table: the same file can have several roles and references.
Do not reduce a record to whichever role was loaded last.

| Source | Existing inputs to reuse | Required treatment |
| --- | --- | --- |
| PlantVillage | `artifacts/efficientnet_b0/{train,validation,test}_manifest.csv`; corresponding robust manifests; C1 saved mixed-source manifests | Original split counts 2,214/475/475. Check reference/robust inventory identity. Include original Apple colour files and known aliases, mapping exact-content aliases to all applicable roles. PV `image_hash` is a legacy pixel hash, not a raw-file hash. |
| PP2020 | `data/field_manifests/pp2020_audit/{pp2020_inventory,previous_use_registry}.csv`; duplicate-adjudication group members/constraints; role-preservation registries | Cover all 1,821 labelled originals and all 2,533 registered file members, including excluded records, copies, quarantine and reservations. Retain group constraints and all prior-use evidence. |
| PP2020 frozen C1 roles | `data/field_manifests/pp2020_split_v1/{train,val,test,excluded}.csv` and reports; `artifacts/efficientnet_b0_field_c1_v1/{train_manifest,plantvillage_validation_manifest,pp2020_validation_manifest}.csv` | Actual PP2020 membership is 1,176 train, 251 validation, 252 protected test. Overlay these roles onto the earlier registries. C1 validation guided selection; C1 test predictions now exist. Neither validation nor test aliases may enter C2 training. |
| Zenodo real lesions | `lesion_manifests/{train,train_before_hard_negatives,val,test}.csv`; restricted dataset image folders | 30 real training, 15 validation, 30 test images. Register their actual roles and aliases. Masks are auxiliary labels, not disease-free photographs; preserve and link them without adding them as ordinary photo nodes. |
| Zenodo synthetic lesions | Current training manifests; `data/lesion_zenodo/synthetic/Synthetic_Dataset/{stable_diffusion_images,blender_renders}` | Register both photograph/render representations, including files outside the active training manifest. Current training includes 300 synthetic images. Record generator/source lineage if available; missing lineage is not proof of independence. Exclude `masks/` from the photo-matching index while recording its auxiliary linkage. |
| PP2020 healthy negatives/reservations | `lesion_manifests/train.csv`; `data/healthy_structural_benchmark/manifest.csv`; `pp2020_role_preservation/{protected_heldout_members,segmentation_training_classification_reconciliation}.csv` | Current lesion training has 40 hard negatives in addition to 330 real/synthetic rows. Preserve the 50 evaluation-only healthy groups and 100 registered members. Segmentation-trained images are not unseen combined-pipeline evaluation examples. |
| Kashmir | Original evaluation manifests, e.g. `artifacts/kashmir_evaluation/efficientnet_b0/manifest.csv`; recovered visual manifest; `results/2026-10-08_kashmir_label_audit/review_manifest.csv`; recovery records and adjudicated pair reviews | Include original and recovered aliases, all 313 readable recovered records and the historical 260-image evaluation membership. Do not register Git LFS pointer text as an image. Preserve 25 conflicted rows and same-label copies, unverified diagnoses and historical evaluation exposure. |

**Important:** the PP2020 role-preservation registry predates frozen C1 splits.
Its empty `assigned_classification_split` and prospective `eligible_*` flags are
not authorization for C2. Frozen roles plus actual training/selection/evaluation
evidence must override those earlier prospective permissions. File copies alone
also do not prove training or selection; preserve the Cedar review's bounded
evidence rather than inventing use from filenames.

Enumerate referenced image folders and reconcile them with manifests. Missing
required sources, unreadable files, unresolved roles or unaccounted files must
produce a coverage error/hold, not a silently incomplete clean audit.

## Hash-method compatibility: a required bridge

The inspected code uses different methods even where the column names agree:

| Source/tool | Pixel digest input | Orientation |
| --- | --- | --- |
| PP2021 `audit_pp2021_integrity.py` | `f'{width}x{height}|'.encode() + rgb.tobytes()` | EXIF transpose then RGB |
| PP2020 `inspect_image` | `str(rgb.size).encode() + rgb.tobytes()` | EXIF transpose then RGB |
| PV `appleleaf.data.image_hash` | `str(rgb.size).encode() + rgb.tobytes()` | RGB conversion without EXIF transpose |
| Kashmir `audit_kashmir_images.py` | `str(rgb.size).encode() + rgb.tobytes()` | RGB conversion without EXIF transpose |

SHA-256 digests cannot be converted between these schemes without the original
pixel bytes. Do not compare incompatible digests or infer non-overlap from a
failed comparison. Raw-file SHA-256 is directly comparable when it represents
actual image bytes, not a Git LFS pointer.

Use PP2021's saved EXIF-oriented RGB method as the cross-dataset canonical pixel
method (`rgb_exif_wh_pipe_sha256_v1`). Reuse all saved PP2021 raw/pixel/dHash
values; no wholesale PP2021 decoding is needed. For reference datasets, reuse
raw digests where trustworthy, and decode each distinct reference file content
once to compute the compatible pixel digest and dHash. Retain the legacy hashes
alongside the new scheme-tagged values and verify their documented methods.
Reuse raw-identical reference cache entries across aliases, retaining all roles.

Bind each cache entry to source raw SHA, method ID, decoder/Pillow version,
oriented dimensions and evidence provenance. Invalidate changed, missing or
method-unknown entries. The initial PP2021 inventory has no per-image original
stat snapshot: C2-B proves preservation during that operation, not a fresh
content-level re-audit. If C2-C cannot establish inventory freshness from trusted
evidence, stream-verify saved raw-file digests as a separate, documented pass;
do not decode everything merely to repeat already valid hash calculations or
pretend file-stat checks establish byte identity.

The PP2021 256-bit dHash shifts 256 row-major `right > left` bits into a big-endian
integer. PP2020 uses `np.packbits` on the same comparisons after EXIF-oriented
RGB -> L and a 17x16 LANCZOS resize. These implementations are compatible by
algorithm, subject to decoder/resampling verification. Check this with synthetic
fixtures and raw-identical reference/PP2021 pairs where available.
The healthy-benchmark helper lacks the same explicit RGB/EXIF normalization, so
wrap/recompute it through the canonical method when necessary.

Kashmir's `find_kashmir_label_conflicts.py` uses **64-bit DCT pHash**. Its distance
and threshold cannot be reused as dHash256 values. Preserve its adjudicated
decisions as evidence, and compute canonical dHash256 for registry photos.

## Efficient matching, complete pair coverage and review

1. Create inverted dictionaries from raw SHA and compatible pixel SHA to all
   reference node IDs. Probe each PP2021 record, retaining every alias/role
   match, distinguishing raw-byte identity from normalized-pixel identity.
2. Index unique compatible reference dHash256 values using **five disjoint bit
   blocks** (52, 51, 51, 51, 51 bits). For each unique PP2021 dHash, union references
   sharing an identical block in the same block position, then verify full
   `(left ^ right).bit_count() <= 4`. At distance <=4 at least one of five blocks
   must be unchanged, so this indexing preserves threshold completeness.
   Expand content nodes to all record aliases after matching and deduplicate
   emitted pair IDs deterministically.
3. This avoids an unconditional PP2021-by-reference all-pairs scan. Record index
   sizes, candidates inspected, distances and any high-collision buckets. Do not
   silently cap buckets or truncate pairs; if resources prevent completeness,
   mark the audit incomplete and stop. A tested exact BK-tree is an alternative
   if real bucket collisions make the five-block index ineffective.
4. Also screen PP2021-to-PP2021 near-duplicate candidates with the same indexed
   method, using all 18,632 records, excluding self/reversed pairs. The earlier
   audit established exact duplicates only; excluded PP2021 images may bridge
   other PP2021 nodes to protected sources through reviewed near-duplicate links.
5. Generate paginated contact sheets with stable pair IDs, original labels,
   paths, source roles, raw/pixel equality, dHash distance and review status.
   Produce a review template requiring reviewer, decision, reason, and evidence.
   Allowed decisions: `confirmed_duplicate`, `distinct_photographs`, `unresolved`.
   Near-identical crops or related views may require acquisition/leaf evidence;
   resemblance alone does not settle biological independence.
6. dHash <=4 is **screening only**. Never relabel, permanently reject, or union
   visually similar photos automatically. Unreviewed candidates remain a
   documented review hold where they could affect eligibility. A distance >4
   is outside this initial screen, not proof of independence; crops, rotations,
   compression and related views can evade it.

## Confirmed components and restriction propagation

Seed confirmed components with saved PP2021 exact groups and all adjudicated
reference groups. Add raw/pixel exact edges only when hash methods and source
integrity are verified; exact content establishes identity, not which disease
label is correct. Add perceptual edges only after explicit confirmation. Use
union-find over dataset-qualified nodes, not filenames alone, and propagate
restrictions transitively across all members, including excluded records.

Treat permission as the **intersection of all applicable restrictions**, not a
single role that can erase another. Export all flags plus the dominant hold:

- Existing C2-B conflicts stay `quarantined_label_conflict`; new incompatible
  source labels or unresolved diagnostic equivalence stay quarantined/pending.
- Any confirmed link to protected PV/PP2020/lesion validation or test, healthy
  reserved evaluation, or Kashmir external evaluation forbids C2 training.
  Historical validation/selection/evaluation data cannot become new unseen tests.
- Any confirmed previous training link forbids independent unseen evaluation
  for the corresponding classifier or combined pipeline. Prior segmenter-only
  exposure remains separately recorded; it is not hidden by classifier novelty.
- A component containing both prior training and protected held-out roles is a
  prior-role conflict: hold it and flag the earlier evaluation-integrity concern.
  Neither role overwrites the other and no frozen assignment is changed.
- Unreviewed near-duplicate/role/integrity questions produce a temporary hold,
  not a diagnosis or an automatic duplicate rejection.
- No overlap found means `no_detected_overlap_pending_protocol_and_label_review`,
  not proof of independence or automatic training permission. C2-C still assigns
  no splits and cannot clear provisional Frogeye diagnoses on its own.

## Proposed auditable output contract

Write a new directory, e.g.
`data/field_manifests/pp2021_cross_dataset_v1/`; identical verified outputs may be
reused, and changed/incomplete outputs require a new explicit version. Stage
files together, freeze input/script/method hashes, and never rewrite old audits.

| Proposed output | Purpose / essential fields |
| --- | --- |
| `reference_registry.csv` | Qualified node ID, dataset, original ID/path, labels/mapping state, source hashes/methods, all roles, group IDs |
| `role_evidence.csv` | Node ID, role, experiment/model scope, manifest path/row/hash, evidence status |
| `hash_compatibility_report.json` | Method definitions, bridges/caches, orientation/resampling tests, versions, freshness/coverage status |
| `reference_hash_cache.csv` | Source SHA, canonical pixel SHA/dHash, method IDs, decoder version, dimensions, cache provenance |
| `exact_matches.csv` | PP2021 ID, reference ID or PP2021 peer, raw/pixel match flags, method IDs, labels and role evidence |
| `perceptual_candidates.csv` | Stable pair ID, source IDs, dHash256 values/distance, original labels/roles, exact flags, pending-review state |
| `review_decisions_template.csv` | Pair ID and image hashes, reviewer, decision, reason, evidence, review date; append decisions as a separately versioned artifact |
| `review_sheets/` and `review_sheet_index.csv` | Visual pair evidence with stable file/page/tile references |
| `confirmed_group_members.csv` | Every node, confirmed group, edge-evidence provenance; no unreviewed perceptual union |
| `group_restrictions.csv` | Group ID, all inherited roles/flags, dominant hold/reason, supporting nodes, forbidden uses |
| `pp2021_overlap_status.csv` | Exactly 18,632 original rows with original labels, proposed labels, C2-B state, matches/holds and restrictions |
| `eligibility_summary.json` | Original/remaining class reconciliation, quarantined/protected/previously-trained/review-pending counts, coverage errors, zero automatic approvals/splits |
| `cross_dataset_integrity_report.json` | Source/output hashes, role coverage, index completeness counters, preservation checks, exclusions, limitations, pass/incomplete status |

All 14,491 original candidates, the 27 removed candidates, and the 14,464 pending
ones must reconcile in the summary. Cross-dataset groups may change their
restrictions, but must not modify the frozen C2-B files or original source labels.

## Existing reusable code and what must change

- `scripts/audit_pp2020_for_field_training.py`: streaming `sha256`,
  `inspect_image` normalization/dHash, prior-use manifest validation. Its pixel
  header needs an explicit canonical bridge. Its nested pairwise screening is
  suitable as a small-fixture oracle, **not** the PP2021 search implementation.
- `scripts/adjudicate_pp2020_duplicates.py`: `Components`, `judge_pair`,
  contact-sheet conventions and confirmed-link constraints. Adapt the fixed
  PP2020 label-signature policy; do not reuse it unchanged for PP2021 labels.
- `scripts/preserve_pp2020_heldout.py`: `group_permissions` and
  `validate_proposed_roles` for strongest-use propagation and alias protection.
  Extend policies for frozen classifier validation/test and current C1 history.
- `scripts/build_pp2020_field_splits.py --verify`: read-only frozen-version,
  input-hash, deterministic-replay and prior-role checks. Reuse verification;
  never call its split-building path for C2.
- `scripts/review_pp2020_cedar_usage.py`: evidence distinction between copies,
  completed training, selection, actual predictions and unconfirmed membership.
- `scripts/build_healthy_benchmark.py`: row-major dHash/Hamming helper and
  reserved benchmark metadata; normalize orientation/RGB explicitly.
- Kashmir recovery, review manifests and adjudicated cross-folder pair CSVs:
  reuse provenance and decisions, not unverified diagnosis labels or incompatible
  pHash distances. `find_kashmir_label_conflicts.py` has top-level side effects;
  do not import it as a utility.
- `scripts/audit_pp2021_integrity.py` also executes its initial audit at module
  level. Do not import or rerun it: reuse its saved inventory and implement pure,
  tested compatible helpers without triggering a second 18,632-image audit.

## Implementation order, tests and exact next step

**Next implementation step:** create `scripts/audit_pp2021_cross_dataset.py`
and `tests/test_pp2021_cross_dataset.py`. Start with read-only registry adapters
and a `--validate-only` command that verifies coverage, C2-B provenance, frozen
roles and method compatibility before matching. No training/model loader belongs
in this script.

Tests must cover raw-identical renamed aliases, pixel-identical files with
different metadata bytes, EXIF rotations, same pixels under incompatible legacy
headers, dHash bit ordering and distance boundaries 0/4/5, indexed-vs-brute-force
pair equality on small fixtures, alias expansion, PP2021 excluded-node bridges,
transitive strongest restrictions, C1 validation/test protection overriding old
prospective registries, prior training/held-out conflicts, no automatic union of
perceptual candidates, all-18,632-row coverage and fail-closed incomplete sources.
Also test preservation, deterministic output, cache invalidation and no-overwrite.

After implementation and tests, the intended first command is:

```powershell
.venv\Scripts\python.exe scripts/audit_pp2021_cross_dataset.py --validate-only --quarantine-dir data/field_manifests/pp2021_audit/internal_quarantine_v1 --output-dir data/field_manifests/pp2021_cross_dataset_v1
```

This is the **planned interface**, not a currently executable command: the
cross-dataset script has deliberately not been created in C2-B. Only after its
validation passes should the matching-only mode run, followed by visual review
and restriction adjudication. Split design and classifier training remain
separate future tasks.
