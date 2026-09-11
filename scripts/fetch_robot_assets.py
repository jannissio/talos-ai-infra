"""Download only the licensed, pinned Menagerie SO-101 assets used by this sandbox."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

COMMIT = "ac6b2b09983786f3036cab1000221017fa2193b4"
SOURCE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/robotstudio_so101"
DESTINATION = Path(__file__).resolve().parents[1] / "simulation_lab" / "assets" / "so101"


def fetch(relative: str) -> dict:
    path = DESTINATION / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SOURCE}/{relative}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    path.write_bytes(content)
    return {"file": relative, "source": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main() -> None:
    records = [fetch("so101.xml"), fetch("LICENSE"), fetch("README.md")]
    tree = ET.parse(DESTINATION / "so101.xml")
    for mesh in tree.findall("./asset/mesh"):
        records.append(fetch(f"assets/{mesh.attrib['file']}"))
    (DESTINATION / "provenance.json").write_text(json.dumps({"commit": COMMIT, "license": "Apache-2.0", "files": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} verified source files ({sum(r['bytes'] for r in records):,} bytes) to {DESTINATION}")


if __name__ == "__main__":
    main()
