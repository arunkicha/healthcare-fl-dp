import hashlib
from pathlib import Path

folder = Path("data/raw/mimic-iii-clinical-database-1.4")
checksums = {}
for line in (folder / "SHA256SUMS.txt").read_text().splitlines():
    h, fname = line.split()
    checksums[fname] = h

for fname, expected in checksums.items():
    path = folder / fname
    if not path.exists():
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    print(f"{fname}: {'OK' if actual == expected else 'MISMATCH!'}")