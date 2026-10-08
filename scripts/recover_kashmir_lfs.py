"""Recover exact Git LFS objects into a separate dataset; preserve the original."""
import argparse
import hashlib
import io
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/external_classification/kashmir"))
    parser.add_argument("--destination", type=Path, default=Path("data/external_classification/kashmir_recovered"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/kashmir_label_audit/recovery"))
    args = parser.parse_args()
    if args.destination.exists() or args.output_dir.exists():
        raise ValueError("Use new destination and output directories")
    pointers = []
    for path in sorted(args.source.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
            match = re.search(rb"oid sha256:([a-f0-9]{64})\s+size (\d+)", payload)
            if not match:
                raise ValueError(f"Malformed pointer: {path}")
            pointers.append({"relative_path": path.relative_to(args.source).as_posix(),
                             "oid": match[1].decode(), "size": int(match[2]),
                             "pointer_sha256": hashlib.sha256(payload).hexdigest()})
    args.output_dir.mkdir(parents=True)
    shutil.copytree(args.source, args.destination)
    batch_url = "https://github.com/bhataakib02/Apple-Disease-Dataset.git/info/lfs/objects/batch"
    response = requests.post(batch_url, json={"operation": "download", "transfers": ["basic"],
        "objects": [{"oid": p["oid"], "size": p["size"]} for p in pointers]},
        headers={"Accept": "application/vnd.git-lfs+json"}, timeout=60)
    response.raise_for_status()
    objects = {obj["oid"]: obj for obj in response.json()["objects"]}

    def recover(row):
        record = dict(row)
        try:
            obj = objects[row["oid"]]
            if "error" in obj:
                raise ValueError(obj["error"])
            action = obj["actions"]["download"]
            downloaded = requests.get(action["href"], headers=action.get("header", {}), timeout=60)
            downloaded.raise_for_status()
            payload = downloaded.content
            if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["oid"]:
                raise ValueError("Downloaded size or SHA-256 does not match original pointer")
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.convert("RGB").load()
                record["width"], record["height"] = image.size
            target = args.destination / row["relative_path"]
            target.write_bytes(payload)
            record["status"] = "recovered_size_hash_and_image_verified"
        except (requests.RequestException, ValueError, KeyError, OSError) as error:
            record["status"] = "failed"
            record["error"] = str(error)
        return record

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(recover, pointers))
    report = {"source": str(args.source.resolve()), "destination": str(args.destination.resolve()),
              "lfs_repository": "https://github.com/bhataakib02/Apple-Disease-Dataset",
              "pointer_count": len(pointers), "recovered_count": sum(r["status"].startswith("recovered") for r in records),
              "records": records}
    (args.output_dir / "recovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Recovered {report['recovered_count']}/{report['pointer_count']} exact images; originals preserved", flush=True)
    if report["recovered_count"] != report["pointer_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
