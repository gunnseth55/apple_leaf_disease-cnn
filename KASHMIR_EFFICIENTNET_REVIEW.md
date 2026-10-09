# Frozen EfficientNet-B0: Kashmir class and visual review

Reviewed 2026-10-10 using saved predictions only. **EfficientNet-B0 remains the
current best PlantVillage classifier, frozen at epoch 11.** No weights, source
labels, thresholds, preprocessing or evaluation exclusions were changed.

Checkpoint: `artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt`.
SHA-256 before and after this review:
`d6b94fa0b61250b28930e92e1c676fce89b6575204ba510b3258f521f53d078c`.
Selection remains based on PlantVillage validation macro-F1 (1.0000), not Kashmir.

## Preliminary class breakdown

**Overall Kashmir accuracy 26.15% (68/260) and present-class macro-F1 0.2248 are
preliminary.** Rot-to-black-rot mapping and source disease labels remain
unverified. Independent verification is required before a definitive score.
This run uses the original 260-image available subset, not the recovered
283-image candidate inventory. Six exact-conflict rows and 47 original LFS
pointers were excluded; 16 evaluated rows still belong to the later audit's
visually confirmed conflicting-label groups. Do not reinterpret these scores
as a cleaned benchmark or as accuracy against confirmed diagnoses.

| Source folder label | Support | Predicted scab | Predicted black rot | Predicted cedar rust | Predicted healthy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scab | 117 | 32 (27.35%) | 2 (1.71%) | 1 (0.85%) | 82 (70.09%) |
| Rot (black-rot mapping provisional) | 100 | 44 (44.00%) | 2 (2.00%) | 0 | 54 (54.00%) |
| Healthy | 43 | 9 (20.93%) | 0 | 0 | 34 (79.07%) |
| Total | 260 | 85 | 4 | 1 | 170 |

Scab recall is **32/117 = 27.35%**; most scab-folder images are classified
healthy. Healthy recall is **34/43 = 79.07%**, but healthy precision is only
**20.00%**: 136 of the 170 healthy predictions come from diseased folders.
High healthy recall therefore does not establish safe disease screening.

The rot folder is mostly classified healthy or scab. Its 2.00% agreement with
black rot is conditional on an unverified mapping; it cannot establish that
the folder contains black rot or that the model has failed against confirmed
black-rot diagnoses. The observed disagreements are retained without relabeling.

Cedar rust is predicted **1/260 times (0.38%)**, exclusively on a scab-folder
image (`SCAB LEAVES/4086.jpg.jpeg`, review ID K266). No cedar-rust ground truth
exists. Its recall is unmeasured, rather than demonstrated to be zero.

## Representative visual inspection

Inspected all 12 examples on two contact sheets: 11 folder-label disagreements
plus one rot/black-rot agreement for context. Selection covers the main
folder/prediction combinations at highest, middle-ranked and selected lowest
confidence, rather than only the globally most confident errors. These examples
illustrate failure patterns; they are not a random sample or a disease diagnosis.

| Examples | Pattern | Visual observations |
| --- | --- | --- |
| E01, E03 | Scab to healthy | Visible reddish-brown/tan patches; E01 is predicted healthy at 0.9999 confidence. E03 is soft-focus and lower-confidence. |
| E02 | Scab to healthy | Pale underside and small dark marks; no large lesion apparent at contact-sheet scale. Folder diagnosis cannot be established visually. |
| E04, E05 | Rot to healthy | Dark blades on pink backgrounds; visible texture/discoloration varies. E04 has 0.999985 healthy confidence. Neither health nor black rot is established from these photos. |
| E06, E07 | Rot to scab | Irregular torn/missing tissue, holes and brown margins. E07 belongs to the previously recorded rot/scab conflict G09. |
| E08, E09 | Healthy to scab | Green/pale leaves with prominent veins on textured fabric; no obvious large lesions at sheet scale. |
| E10 | Scab to black rot | Numerous pale/yellow marks on the underside; the observed pattern does not establish either disease. |
| E11 | Scab to cedar rust | Broad brown patch and soft focus; maximum probability only 0.3332. This is a weak four-way argmax, not a confident rust diagnosis. |
| E12 | Rot to black rot (agreement) | Broad brown tissue and torn/missing right-edge tissue. Agreement with the provisional label does not confirm disease identity. |

[Sheet 1](artifacts/kashmir_evaluation/efficientnet_b0_review/representative_sheet_01.jpg)
and [sheet 2](artifacts/kashmir_evaluation/efficientnet_b0_review/representative_sheet_02.jpg)
show the original photos, prediction, confidence and audit-conflict flags.
Per-example paths, probabilities, review IDs and individual visual notes are in
`results/2026-10-10_efficientnet_b0/kashmir_review/representative_examples.csv`.

Pink/blue backgrounds, underside views, texture, illumination, blur and visible
damage vary across these examples. These are observed differences, not proof
of the cause of prediction failure. No Grad-CAM or intervention was run here.
Visual observations must not be substituted for expert diagnostic labels.

## Confidence and decision

Across the subset, mean confidence is 0.8440. Errors have higher mean confidence
(0.8534) than folder-label agreements (0.8175). Of 141 predictions with maximum
softmax >=0.90, 107 disagree with source labels (75.89%). This supports treating
softmax confidence as uncalibrated, not as a reliability guarantee.

Keep EfficientNet-B0 frozen as the current classifier reference. Its PlantVillage
performance is strong, while this preliminary external assessment does not
validate reliable diagnosis. Verify rot identity and conflicting assignments
before presenting a definitive total Kashmir score; do not train, retune,
choose thresholds or correct labels using this evaluation set.

Reproduce the saved-output breakdown/contact sheets with:

```powershell
.venv/Scripts/python.exe scripts/review_kashmir_efficientnet.py
```

This script loads saved predictions and images, not model weights. It also
records this session's manual visual observations; it does not automate diagnosis.
It verifies checkpoint hash preservation. Machine-readable breakdown and review
summary are in `results/2026-10-10_efficientnet_b0/kashmir_review/`.
