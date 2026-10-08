import re
from pathlib import Path

out = Path("artifacts/kashmir_label_audit/source")
s = (out / "mirror_raw_tree.html").read_text(encoding="utf-8")
print("Mirror paths:", sorted(set(re.findall(r'"path":"([^"]+)"', s))))
s = (out / "drive_folder.html").read_text(encoding="utf-8")
for match in re.findall(r'.{0,100}(?:APPLE ROT|SCAB LEAVES|HEALTHY LEAVES|LEAF BLOTCH).{0,200}', s)[:12]:
    print(match)
