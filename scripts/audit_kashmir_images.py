"""Inventory recovered Kashmir images and make numbered visual review sheets."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    root = Path("data/external_classification/kashmir_recovered")
    out = Path("artifacts/kashmir_label_audit/visual")
    out.mkdir(parents=True, exist_ok=False)
    recovery = json.loads(Path("artifacts/kashmir_label_audit/recovery/recovery.json").read_text())
    restored = {r["relative_path"] for r in recovery["records"] if r["status"].startswith("recovered")}
    rows, bad = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                width, height = rgb.size
            rows.append({"review_id": f"K{len(rows)+1:03}", "relative_path": relative,
                         "folder_label": path.parent.name, "restored": relative in restored,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "pixel_sha256": pixel_hash, "width": width, "height": height})
        except (OSError, ValueError) as error:
            bad.append({"relative_path": relative, "error": str(error)})
    duplicate_reports = {}
    for hash_name in ("sha256", "pixel_sha256"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[hash_name]].append(row)
        groups = [group for group in grouped.values() if len(group) > 1]
        conflicts = [group for group in groups if len({r["folder_label"] for r in group}) > 1]
        duplicate_reports[hash_name] = {"duplicate_groups": groups, "conflicting_groups": conflicts}
    conflict_ids = {r["review_id"] for group in duplicate_reports["pixel_sha256"]["conflicting_groups"] for r in group}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
    for folder in sorted({r["folder_label"] for r in rows}):
        selected = [r for r in rows if r["folder_label"] == folder]
        for start in range(0, len(selected), 24):
            page = selected[start:start+24]
            sheet_name = f"{folder.lower().replace(' ', '_')}_{start//24+1:02}.jpg"
            canvas = Image.new("RGB", (1440, 1100), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 10), f"Folder label: {folder} | page {start//24+1} | R = recovered, C = exact label conflict", font=title_font, fill="black")
            for index, row in enumerate(page):
                x, y = (index % 6)*240, 50+(index//6)*260
                with Image.open(root / row["relative_path"]) as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"), (230, 205))
                canvas.paste(thumbnail, (x+(240-thumbnail.width)//2, y))
                flags = (" R" if row["restored"] else "") + (" C" if row["review_id"] in conflict_ids else "")
                filename = Path(row["relative_path"]).name
                caption = f"{row['review_id']}{flags} {filename}"
                draw.text((x+4,y+208), caption[:31], font=font, fill="red" if flags else "black")
                draw.text((x+4,y+228), caption[31:62], font=font, fill="black")
                row["review_sheet"] = sheet_name
            canvas.save(out / sheet_name, quality=95)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"total_files": len(rows)+len(bad), "readable_images": len(rows),
               "unreadable": bad, "folder_counts": dict(Counter(r["folder_label"] for r in rows)),
               "restored_counts": dict(Counter(r["folder_label"] for r in rows if r["restored"])),
               "duplicates": duplicate_reports, "conflicting_rows": len(conflict_ids),
               "eligible_after_exact_conflict_exclusion": len(rows)-len(conflict_ids),
               "rot_label_status": "unconfirmed; source folder name is not a pathogen diagnosis"}
    (out / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in summary.items() if key != "duplicates"}, indent=2))
    print("Conflict IDs:", sorted(conflict_ids))


if __name__ == "__main__":
    main()
