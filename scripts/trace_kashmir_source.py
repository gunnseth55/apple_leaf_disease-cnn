"""Fetch public provenance metadata without executing downloaded code."""
from pathlib import Path
import requests

out = Path("artifacts/kashmir_label_audit/source")
out.mkdir(parents=True, exist_ok=True)
urls = {
    "mirror_canonicalize.py.txt": "https://raw.githubusercontent.com/bhataakib02/Apple-Disease-Dataset/main/canonicalize_classes.py",
    "mirror_setup.py.txt": "https://raw.githubusercontent.com/bhataakib02/Apple-Disease-Dataset/main/setup_datasets.py",
    "mirror_raw_tree.html": "https://github.com/bhataakib02/Apple-Disease-Dataset/tree/main/raw",
    "drive_folder.html": "https://drive.google.com/drive/folders/1FWsZxnEGdcUfMjhAXwCP4KKkpViABrGI",
    "mirror_kashmir_tree.html": "https://github.com/bhataakib02/Apple-Disease-Dataset/tree/main/raw/kashmir",
    "mirror_sample_pointer.txt": "https://raw.githubusercontent.com/bhataakib02/Apple-Disease-Dataset/main/raw/kashmir/APPLE%20ROT%20LEAVES/5062.jpg%281%29.jpeg",
    "drive_rot_folder.html": "https://drive.google.com/drive/folders/1aPsRF29ZGVo8xtkbgdoIXcqszCp9nS92",
    "drive_scab_folder.html": "https://drive.google.com/drive/folders/1bOhzYPxK-iO7i0lAJ9rOsYn89DjPVRx7",
}
for name, url in urls.items():
    if (out / name).exists():
        continue
    try:
        response = requests.get(url, timeout=45)
        print(name, response.status_code, len(response.content), flush=True)
        if response.ok:
            (out / name).write_bytes(response.content)
    except requests.RequestException as error:
        print(name, str(error), flush=True)

url = "https://github.com/bhataakib02/Apple-Disease-Dataset.git/info/lfs/objects/batch"
response = requests.post(url, json={"operation": "download", "transfers": ["basic"],
    "objects": [{"oid": "e6e2099c925cfbcedf7c5f6a73d9c1cefa198c48a9f9d47f698e65300da1821b", "size": 38569}]},
    headers={"Accept": "application/vnd.git-lfs+json"}, timeout=45)
print("LFS sample", response.status_code, flush=True)
(out / "lfs_sample_batch.json").write_text(response.text, encoding="utf-8")
if response.ok:
    obj = response.json().get("objects", [{}])[0]
    print("LFS object has download action:", "download" in obj.get("actions", {}), "error:", obj.get("error"), flush=True)
