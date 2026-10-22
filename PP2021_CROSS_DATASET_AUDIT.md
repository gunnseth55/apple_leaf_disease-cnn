# C2-C: PP2021 cross-dataset matching

Run date: 2026-10-23. This is a data-identity audit, with no model loading,
inference, classifier training, diagnostic relabeling or C2 split assignment.

## Outcome

**Matching is complete; visual review is still required.** No cross-dataset
raw-file or compatible normalized-pixel duplicates were found. The 27 existing
internal PP2021 exact pairs were reproduced and remain quarantined.

The radius-four dHash screen found **1,293 nonexact pairs**, all between PP2021
and PP2020, involving **706 PP2021 records** and **1,282 PP2020 paths**. Aliases
are retained as separate path-level evidence, so pair counts are not counts of
independent photographs. All 1,293 new decisions remain unresolved; there are
no newly confirmed visual duplicates or automatic duplicate rejections.

| dHash Hamming distance | Path-level candidate pairs |
| --- | ---: |
| 0 | 865 |
| 1 | 335 |
| 2 | 83 |
| 3 | 8 |
| 4 | 2 |
| **Total** | **1,293** |

A zero dHash distance is not an exact full-pixel match. The matching image may
have different dimensions, compression, or other differences. The absence of
raw/pixel equality must not be used to clear perceptual candidates for training.
No additional internal PP2021 nonexact radius-four pairs were found; the
screen also found no radius-four PP2021 pairs against the other reference sets.
These are results within the tested methods and threshold, not proof of independence.

### Effect on the 14,464 C2-B remaining candidates

| Current state | Candidate records |
| --- | ---: |
| No detected overlap; still pending protocol/label review | 10,688 |
| Frogeye diagnostic mapping review pending | 3,181 |
| Visual duplicate review pending | 595 |
| **Total retained, not approved for training** | **14,464** |

The 595 visual holds comprise **594 Rust and one Healthy** candidate. Of them,
**176** have possible links to protected PP2020 classifier roles: 88 to test and
88 to validation. Another 418 have possible prior-training links; one has other
unresolved source-role evidence. These are possible relationships, not confirmed
restrictions inherited across a speculative edge. All candidates remain
ineligible for training until the appropriate reviews and later protocol steps.

Across the full 18,632-row inventory, 684 records have a visual-review status,
54 retain label-conflict quarantine, 3,181 need diagnostic mapping review, and
14,713 have no detected overlap but remain pending. Of the 706 records appearing
in perceptual pairs, 22 are already in the 54-record quarantine and retain that
stronger state. Excluded original labels remain excluded through `c2b_status`;
matching them does not add them to the single-label candidate pool.

### Verification and preservation

**All 79 repository tests passed**, including 14 new audit tests and two tests
for the independent verifier. The complete saved audit also passed an independent
graph reconstruction and an alternate eight-block dHash index: exact and
perceptual pair sets matched, with **zero unsupported confirmed unions**.
Every original PP2021 label and saved identity was reconciled; all 54 original
quarantine records stayed quarantined. Review-template and contact-sheet
coverage matched every pending pair.

The main cross-dataset index performed only **703 full Hamming comparisons**
on unique hash candidates, before alias expansion, versus 129,641,456 possible
path-level cross-dataset pairs in an unconditional scan. Internal screening made
18,617 comparisons including self-hash queries. Neither scan truncated results.

Before/after preservation checks found no change to **421 metadata/weight files**
or file metadata for **26,282 workspace image/archive paths**. All 25,590 indexed
photographs, including cached PV paths outside the workspace, were also checked
through their raw-freshness pass and unchanged per-node file metadata. No weights,
source images, original labels, C2-B files or frozen splits were rewritten.

## Review handoff

The full audit is in `data/field_manifests/pp2021_cross_dataset_v1/`. It includes
**216 contact sheets** covering every candidate pair. A sheet was inspected for
layout; this is not a claim that all pairs have been adjudicated.

A compact verified snapshot and group-aware queue are in
`results/2026-10-23_pp2021_cross_dataset/`:

- [priority_review_queue.csv](results/2026-10-23_pp2021_cross_dataset/priority_review_queue.csv): all pair IDs, source labels, dimensions-independent hash evidence, inherited reference-group roles, review priority and contact-sheet links.
- [review_priority_summary.json](results/2026-10-23_pp2021_cross_dataset/review_priority_summary.json): 356 protected-history path-pairs first, 849 prior-training pairs next, 88 other unresolved pairs; these are not independent-image counts.
- [verification_report.json](results/2026-10-23_pp2021_cross_dataset/verification_report.json): independent completed-audit verification.
- [eligibility_summary.json](results/2026-10-23_pp2021_cross_dataset/eligibility_summary.json): reconciled counts and zero approvals/splits.
- `cross_dataset_integrity_report.json`, `hash_compatibility_report.json`, and
  `snapshot_sha256.json`: preserved run provenance and snapshot checksums.

The queue derives reference roles from **confirmed reference components**, so a
copy cannot hide its original's validation/test role. Its priority is not a
duplicate decision. The next step is to adjudicate the queued pairs, beginning
with protected-history links, recording reviewer/decision/reason/evidence and
then propagating only confirmed links in a new audit version. Do not assign C2
splits or train before resolving the relevant holds and diagnostic mapping.

## Procedure and source coverage

Implementation: [scripts/audit_pp2021_cross_dataset.py](scripts/audit_pp2021_cross_dataset.py).
Independent saved-output verifier:
[scripts/verify_pp2021_cross_dataset.py](scripts/verify_pp2021_cross_dataset.py).

The audit starts from all **18,632** PP2021 inventory rows, including excluded
labels and all 54 C2-B quarantined originals. It verifies the C2-B input/output
hashes and reconstructs the quarantine transformation before matching.

| Reference dataset | Photograph paths |
| --- | ---: |
| PlantVillage original Apple colour images, including deduplicated aliases | 3,171 |
| PP2020 originals and registered copies | 2,533 |
| Zenodo real lesion images | 75 |
| Zenodo synthetic images used in training | 300 |
| Zenodo Blender render counterparts | 300 |
| Kashmir recovered images and readable original aliases | 579 |
| **Total reference paths** | **6,958** |

The registry therefore contains **25,590 photograph paths**. The original PV
2,214/475/475 and PP2020 1,176/251/252 train/validation/test roles are retained.
C1's saved training and validation manifests are checked against those roles.
Historical prospective permissions do not overwrite frozen validation/test
restrictions. Previous segmenter training, lesion validation/test, the protected
healthy benchmark, and Kashmir external evaluation are recorded separately.

The 47 original Kashmir LFS pointers are verified against recovery evidence and
kept outside the photograph index. Masks are linked auxiliary labels, not
healthy photographs. `auxiliary_sources.csv` has 842 evidence rows: repeated
mask-manifest references plus the 47 pointers; this is not a count of independent
images. Original filenames and original labels remain in the registry/evidence.

## Hash compatibility and freshness

The initial PP2021 audit hashes EXIF-oriented RGB bytes with a
`widthxheight|` header. Legacy PV/PP2020/Kashmir pixel hashes use a tuple-size
header, with differing orientation handling. The audit never compares those
incompatible pixel digests directly.

It streams raw-file SHA-256 once for all 25,590 registry photographs and verifies
every available saved raw digest. This freshness pass is necessary because the
initial PP2021 inventory lacks per-image original stat receipts. It reuses all
PP2021 saved pixel/dHash values rather than decoding the entire dataset again.

For references, it decodes **5,889 distinct raw contents**, sharing each result
across raw-identical aliases. It computes a compatible EXIF-oriented RGB
`widthxheight|` SHA-256 and checks the appropriate legacy tuple digest. The
canonical dHash uses grayscale 17x16 LANCZOS, row-major right-greater-than-left
comparisons, and 256 big-endian bits. Saved PP2020 dHashes are checked against
this method; Kashmir's earlier 64-bit DCT pHash is not reused as dHash.

`photo_freshness_receipts.csv` records each verified raw digest and file metadata.
Identical completed audits can be verified/reused, but changed source metadata,
image file metadata, implementation, or output files require a new version.

## Efficient matching and restrictions

Exact checks use inverted raw/pixel SHA dictionaries. The dHash screen uses five
disjoint bit blocks (52/51/51/51/51). At distance <=4, at least one block must be
unchanged. Bucket matches are checked with full 256-bit XOR/popcount. This avoids
an unconditional 18,632-by-6,958 scan and also covers internal PP2021 candidates.
No bucket or pair list is truncated.

Confirmed groups contain only exact-content links and previously adjudicated
source groups. Perceptual candidates are **not** unioned automatically. Prior
training/validation/evaluation restrictions propagate through confirmed groups;
unreviewed near-duplicate links cause review holds. Conflicting source labels
remain unresolved. Every PP2021 record retains its original C2-B state.

## Commands

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts/audit_pp2021_cross_dataset.py --validate-only
.venv\Scripts\python.exe scripts/audit_pp2021_cross_dataset.py --workers 4 --output-dir data/field_manifests/pp2021_cross_dataset_v1
.venv\Scripts\python.exe scripts/verify_pp2021_cross_dataset.py
.venv\Scripts\python.exe scripts/summarize_pp2021_cross_dataset.py
```

Validation mode writes nothing and checks source coverage/provenance. Matching
mode performs raw freshness and reference hash bridging. It stages the complete
audit, checks preservation, and publishes a versioned directory only on success.
An incomplete build is retained under an explicitly marked hidden staging path
for inspection and is never presented as a completed audit.

## Output locations

The versioned audit directory is `data/field_manifests/pp2021_cross_dataset_v1/`:

- `reference_registry.csv`, `role_evidence.csv`: original identities, hashes,
  all roles, source manifests/rows, and pre-existing confirmed-group evidence.
- `auxiliary_sources.csv`: linked masks and LFS pointers outside the photo index.
- `photo_freshness_receipts.csv`, `reference_hash_cache.csv`,
  `hash_compatibility_report.json`: raw freshness and compatible hash-method proof.
- `exact_matches.csv`: byte/pixel exact links, including original internal pairs.
- `perceptual_candidates.csv`: every radius-four nonexact pair with full labels,
  roles, hashes, distances, source paths and review status.
- `review_decisions_template.csv`, `review_sheet_index.csv`, `review_sheets/`:
  stable pair IDs and visual evidence; decisions initially remain unresolved.
- `confirmed_group_members.csv`, `group_restrictions.csv`: confirmed membership
  and inherited restrictions, with no speculative perceptual unions.
- `pp2021_overlap_status.csv`: all 18,632 original PP2021 records and current holds.
- `eligibility_summary.json`, `cross_dataset_integrity_report.json`: reconciliation,
  preservation, source/output hashes, index counters and completion state.

## Interpretation boundaries and next action

An exact identity establishes that content is shared, not which disease label
is correct. dHash proximity identifies potential duplicates; it does not establish
the same photograph or an independently diagnosed disease. Crops, rotations and
related views can evade the initial radius-four screen, so absence of a hit is
not proof of independence.

No candidate is automatically approved for C2 training or unseen evaluation.
Frogeye-to-black-rot remains provisional. Visual adjudication and diagnostic
mapping review precede any future split or training decision. Preserve the
original C2-B audit, checkpoints and frozen splits throughout those later steps.
