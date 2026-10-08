"""Find cross-folder perceptual duplicates for visual review, without relabeling."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn

root = Path("data/external_classification/kashmir_recovered")
out = Path("artifacts/kashmir_label_audit/visual")
with (out / "manifest.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
hashes = {}
for row in rows:
    with Image.open(root / row["relative_path"]) as image:
        pixels = np.asarray(image.convert("L").resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    block = dctn(pixels, type=2)[:8,:8]
    hashes[row["review_id"]] = block > np.median(block)
pairs = []
for i, a in enumerate(rows):
    for b in rows[i+1:]:
        if a["folder_label"] == b["folder_label"] or a["sha256"] == b["sha256"]:
            continue
        distance = int(np.count_nonzero(hashes[a["review_id"]] != hashes[b["review_id"]]))
        same_filename = Path(a["relative_path"]).name == Path(b["relative_path"]).name
        if distance <= 12 or same_filename:
            pairs.append({"a_id": a["review_id"], "b_id": b["review_id"],
                          "a_path": a["relative_path"], "b_path": b["relative_path"],
                          "phash_hamming": distance, "same_filename": same_filename,
                          "status": "candidate_requires_visual_review"})
pairs.sort(key=lambda p:(p["phash_hamming"],p["a_id"],p["b_id"]))
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
for start in range(0,len(pairs),8):
    canvas=Image.new("RGB", (1400,1320), "white")
    draw=ImageDraw.Draw(canvas)
    for k,pair in enumerate(pairs[start:start+8]):
        x,y=(k%2)*700,(k//2)*330
        draw.text((x+5,y+5), f"Candidate P{start+k+1:02}: pHash distance {pair['phash_hamming']}",font=font,fill="black")
        for side,offset in (("a",0),("b",340)):
            with Image.open(root/pair[f"{side}_path"]) as image:
                thumb=ImageOps.contain(image.convert("RGB"),(330,260))
            canvas.paste(thumb,(x+offset+(330-thumb.width)//2,y+30))
            draw.text((x+offset+5,y+294),f"{pair[f'{side}_id']} {Path(pair[f'{side}_path']).name}"[:37],font=font,fill="black")
        pair["pair_id"]=f"P{start+k+1:02}"
        pair["review_sheet"]=f"cross_folder_pairs_{start//8+1:02}.jpg"
    canvas.save(out/f"cross_folder_pairs_{start//8+1:02}.jpg",quality=95)
with (out/"cross_folder_candidates.csv").open("w",newline="",encoding="utf-8") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(pairs[0]))
    writer.writeheader(); writer.writerows(pairs)
print("Cross-folder candidates:",len(pairs))
for pair in pairs:
    print(pair["pair_id"],pair["a_id"],pair["b_id"],pair["phash_hamming"],pair["a_path"],pair["b_path"])
