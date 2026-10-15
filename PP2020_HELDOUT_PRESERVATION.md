# Task C: preserve earlier held-out data

Output directory: `data/field_manifests/pp2020_role_preservation/`.

## Protected and reconciled roles

| Earlier use | Canonical images | Confirmed file copies | Proposed classification permissions |
| --- | ---: | ---: | --- |
| Healthy structural-negative evaluation | 50 | 50 | Existing evaluation only; never new training or validation |
| Segmentation-training hard negatives | 40 | 40 | Classification training or unused; never unseen combined-pipeline validation/test |

Permissions apply to the whole confirmed duplicate component, not filenames.
The protected set therefore includes 100 file members. The 40 previously
trained images and their aliases comprise 80 file members.

Earlier segmentation training does **not** disqualify those 40 healthy images
from classifier training. However, even if a newly trained classifier has never
seen them, the segmentation model has. Performance on them must not be advertised
as unseen evaluation of the combined system. A classifier-only analysis would
require a separate, explicitly scoped protocol; this registry protects the
combined pipeline and allocates these groups to training or unused only.

The existing 50-image benchmark remains an existing, previously inspected
evaluation set. Preserving its separation from training does not make it a newly
unseen test set. Its source manifest and earlier results are unchanged.

## Outputs

- `pp2020_classification_role_registry.csv`: 1,821 original labelled records,
  group IDs, previous-use flags, proposed roles and boolean eligibility fields.
- `pp2020_all_file_role_registry.csv`: the same group permissions extended to
  all 2,533 original/copy paths.
- `protected_heldout_members.csv`: all 100 members of the 50 reserved groups.
- `segmentation_training_classification_reconciliation.csv`: the 40 canonical
  previously trained records, explicitly eligible for classification training.
- `classification_training_candidates.csv`: 1,679 canonical records eligible
  for prospective training allocation (622 rust, 591 scab, 466 healthy).
- `new_pipeline_holdout_candidates.csv`: 1,639 records potentially eligible for
  a future unseen pipeline validation/test allocation.
- `preservation_summary.json`: counts, input hashes and checks.

**Candidate lists are alternative permitted allocations, not actual splits.**
The candidate pools overlap because split assignment has not happened. Do not
train on all candidates and then select evaluation records from the other list.
Choose roles once, by group ID, before training. All assigned split fields remain
empty. The 92 quarantined/excluded records remain blocked under Tasks A/B.

Future split code must call `validate_proposed_roles()` against the all-file
registry. It rejects held-out originals or aliases entering training, previously
trained groups entering unseen pipeline validation/test, unknown paths/groups,
conflicting path/group mappings and duplicate groups spanning active roles.

The earlier audit outputs, healthy evaluation manifest, segmentation training
manifest and source images are preserved. Source hashes were checked against the
audit; input manifest hashes were unchanged before/after this reconciliation.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\preserve_pp2020_heldout.py --output-dir data\field_manifests\pp2020_role_preservation_rerun
```

Use a new/empty output folder. The script validates all 50 reservations and 40
training records against the live manifests, checks canonical source hashes,
and exercises rejection of training assignments for every held-out file alias.
It copies no images, assigns no splits and performs no model training/inference.

All 36 repository tests passed, including four new tests for alias protection,
pipeline novelty, quarantine precedence and proposed-role validation.
