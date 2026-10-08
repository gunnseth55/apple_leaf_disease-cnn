# Kashmir image recovery and label audit

Audit performed on 2026-10-08. **All 47 missing images were recovered, but the
labels are not cleared for a definitive disease benchmark. “Apple Rot” could
not be confirmed as black rot.** Additional contradictory folder labels were
found during visual review.

## Recovery

The missing files were Git LFS pointers. An identical sample pointer was found
in the public
[Apple-Disease-Dataset mirror](https://github.com/bhataakib02/Apple-Disease-Dataset/tree/main/raw/kashmir).
All 47 referenced objects were available through that repository's Git LFS
download endpoint. Every recovered object passed all three checks:

- Byte length matches the original pointer's declared size.
- File SHA-256 matches the original pointer's object ID.
- PIL verification and full RGB decoding succeed.

The original `data/external_classification/kashmir/` remains unchanged.
`data/external_classification/kashmir_recovered/` contains a separate copy with
all 313 files readable. Five recovered rot images are exact duplicates of
existing rot images; the 42 recovered scab images are additional exact-image
content. Recovery of 47 files therefore adds 42 exact-unique images, not 47
independent observations. Related views/recompressed copies remain possible.

| Source folder | All readable files | Recovered files |
| --- | ---: | ---: |
| APPLE ROT LEAVES | 108 | 5 |
| HEALTHY LEAVES | 46 | 0 |
| SCAB LEAVES | 159 | 42 |
| Total | 313 | 47 |

Per-file pointer hashes, object IDs, byte lengths, dimensions, and recovery
status are in [recovery.json](results/2026-10-08_kashmir_label_audit/recovery.json).

### Verification of saved recovery and review outputs

Rechecked the saved outputs on 2026-10-08. All 313 images pass PIL verification
and full RGB decoding, and their file and decoded-pixel hashes match the review
manifest. Each of the 47 recovered files matches the size and SHA-256 declared
by its original LFS pointer; the original pointer hashes also match the recovery
record. The other 266 files are byte-identical between the original and recovered
datasets. The recovered inventory matches the manifest exactly, all referenced
class sheets exist, and the draft status totals and candidate class counts match
the saved review summary. There were no verification errors. Results are saved in
[verification.json](results/2026-10-08_kashmir_label_audit/verification.json).
Representative sheets for rot, healthy, and scab were also reinspected. These
checks confirm recovery and record consistency; they do not confirm diagnoses.

## What “rot” means in the available sources

The mirror's
[download script](https://github.com/bhataakib02/Apple-Disease-Dataset/blob/main/setup_datasets.py)
links its Kashmir data to this
[public Drive folder](https://drive.google.com/drive/folders/1FWsZxnEGdcUfMjhAXwCP4KKkpViABrGI).
The folder listing uses the name `APPLE ROT LEAVES`, and also includes
`LEAF BLOTCH`, which is absent from our local three-folder subset. Neither
folder naming nor successful image recovery verifies disease identity.

The mirror's
[canonicalization code](https://github.com/bhataakib02/Apple-Disease-Dataset/blob/main/canonicalize_classes.py)
explicitly maps `APPLE ROT LEAVES` to `black_rot`. This is a software mapping;
the inspected code provides no diagnostic evidence supporting that equivalence.
It must not be used as independent confirmation of our own provisional mapping.

Related primary research, the
[D-KAP paper, sections IV and V.A](https://d197for5662m48.cloudfront.net/documents/publicationstatus/174054/preprint_pdf/455548e207c392206d1e78b9b233b117.pdf),
describes Kashmir orchard images with apple scab, apple rot, alternaria leaf
blotch, and healthy classes, and explains that labels are inferred from folders.
It does not define the rot class as black rot or provide a pathogen-confirmation
protocol in those sections. Its linked
[Kaggle dataset](https://www.kaggle.com/datasets/hsmcaju/d-kap) is a related
source, not a byte-verified replacement for this exact local collection.

Black rot's leaf manifestation is frogeye leaf spot, commonly showing circular
lesions with tan/brown centers and dark margins. Symptoms can overlap with
other damage, so appearance alone cannot establish that an entire folder has
that diagnosis. See the
[UNH Extension diagnostic reference](https://extension.unh.edu/resource/frogeye-leaf-spot-black-rot-apple-0).
The current evidence supports **unconfirmed**, not confirmed or disproven.

## Visual review of every represented class

All 313 images were screened on 14 numbered contact sheets, including all 47
restored files. This was a visual data-quality review by Codex, not a plant
pathologist's diagnosis. Specific observations for 12 examples are saved in
[example_review.csv](results/2026-10-08_kashmir_label_audit/example_review.csv).

| Folder label | Observations | Assessment |
| --- | --- | --- |
| Rot | Torn/holey blades, marginal browning, large necrotic areas, yellow blades, and some circular spots; repeated views of similar leaves | Heterogeneous damage; black rot not established |
| Healthy | Predominantly green blades, many photographed from below; some holes or small discolored areas | Plausible examples, but no independent health confirmation |
| Scab | Small discrete spots, larger tan lesions, yellowing, and some mildly marked blades; some images also occur under rot | Scab diagnosis not independently verified |

Apple scab is associated with olive-green/brown spots that can become dark and
merge; isolated leaf damage is not enough to validate every source label. See
[University of Minnesota Extension](https://extension.umn.edu/plant-diseases/apple-scab).
No labels were changed based on appearance or model predictions. No cedar-rust
ground truth is present. Leaf blotch was not imported or silently mapped into
the classifier's four classes.

Review sheets are in `artifacts/kashmir_label_audit/visual/`, with each image ID
linked to its path and hashes in `manifest.csv`. Example sheets:

- [Rot](artifacts/kashmir_label_audit/visual/apple_rot_leaves_04.jpg)
- [Healthy](artifacts/kashmir_label_audit/visual/healthy_leaves_01.jpg)
- [Scab, including restored files](artifacts/kashmir_label_audit/visual/scab_leaves_06.jpg)

## Conflicting labels and duplicates

Byte and decoded-pixel hashing recovered the same three exact rot/healthy
conflicts as yesterday: six rows. It also found five extra same-label rot
copies introduced by recovery.

A 64-bit DCT perceptual-hash search (Hamming distance <= 12), plus matching
filenames across folders, generated 84 non-byte-identical cross-folder candidate
pairs. Every candidate was visually reviewed; 12 pairs support eight additional
conflicting groups, involving 19 rows. The other 72 pairs did not establish
duplicates and were not excluded on hash similarity alone.

| Group | Image review IDs | Evidence |
| --- | --- | --- |
| G01 | K001, K129 | Exact rot/healthy duplicate |
| G02 | K002, K145 | Exact rot/healthy duplicate |
| G03 | K003, K151 | Exact rot/healthy duplicate |
| G04 | K017, K176 | Visually same leaf photo, rot/scab |
| G05 | K046, K228, K313 | Near-identical pink-background photo, rot/scab |
| G06 | K047, K229 | Near-identical pink-background photo, rot/scab |
| G07 | K049, K230 | Near-identical pink-background photo, rot/scab |
| G08 | K050, K231 | Near-identical pink-background photo, rot/scab |
| G09 | K051, K232 | Near-identical pink-background photo, rot/scab |
| G10 | K052, K233 | Near-identical pink-background photo, rot/scab |
| G11 | K072, K075, K278, K279 | Near-identical blue-background photo, rot/scab |

This establishes contradictory folder assignments, not which assignment is
correct. Conservative handling excludes every row in these groups until an
authoritative correction is available. This search is not exhaustive: crops,
rotations, compression changes, and related views may evade the threshold.
There are no acquisition or leaf IDs to verify observation independence.

The review decisions are in
[cross_folder_pair_review.csv](results/2026-10-08_kashmir_label_audit/cross_folder_pair_review.csv).
Representative pair comparisons are in
[cross_folder_pairs_02.jpg](artifacts/kashmir_label_audit/visual/cross_folder_pairs_02.jpg).

## Draft usable inventory, not a validated benchmark

After excluding 25 conflicting rows and five same-label duplicate copies,
**283 candidate images** remain: 91 rot, 43 healthy, 149 scab. These are
conservative inventory candidates; their disease labels remain unverified.
The 307 images left after only exact-conflict exclusion would still include
known visual label conflicts and duplicate weighting.

[review_manifest.csv](results/2026-10-08_kashmir_label_audit/review_manifest.csv)
records every row's draft status, conflict group, source label, hashes, and
review sheet. The diagnosis field is deliberately empty. This manifest does
not claim that all remaining labels are correct.

The earlier 24.23% results remain a historical evaluation of the 260-image
subset with provisional mapping. They are not updated scores for the recovered
or audited collection, and the additional conflicts further limit their
interpretation. A new definitive disease score requires a dataset-author or
plant-pathologist label review, especially of rot, scab, and conflicting groups.
That external verification was not performed in this session. No messages were
sent to dataset authors; no classifiers were trained or retuned. Demo integration
remains paused.

## Reproducing the audit

Run the following from the project root. The recovery and inventory scripts
require their destination directories not to exist; keep the saved outputs
for this run or choose new paths before repeating.

```powershell
.venv/Scripts/python.exe scripts/trace_kashmir_source.py
.venv/Scripts/python.exe scripts/recover_kashmir_lfs.py
.venv/Scripts/python.exe scripts/audit_kashmir_images.py
.venv/Scripts/python.exe scripts/find_kashmir_label_conflicts.py
.venv/Scripts/python.exe scripts/record_kashmir_review.py
```

The last script records this session's actual visual decisions, not automated
disease diagnoses. Public source snapshots remain under
`artifacts/kashmir_label_audit/source/`; downloaded source code was inspected as
text and was not executed.
