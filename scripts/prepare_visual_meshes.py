"""Derive lightweight visual meshes; keep every source collision shape unchanged."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import trimesh

ROOT = Path(__file__).resolve().parents[1] / "simulation_lab/assets/so101"


def main():
    source = ET.parse(ROOT / "so101.xml")
    visual = {g.attrib["mesh"] for g in source.findall(".//geom[@class='visual']") if "mass" not in g.attrib}
    collision = {g.attrib["mesh"] for g in source.findall(".//geom") if g.get("class") != "visual" and "mesh" in g.attrib}
    destination = ROOT / "assets/lod"
    destination.mkdir(exist_ok=True)
    records = []
    for node in source.findall("./asset/mesh"):
        filename = node.attrib["file"]
        name = node.get("name", Path(filename).stem)
        if name not in visual or name in collision:
            continue
        mesh = trimesh.load_mesh(ROOT / "assets" / filename)
        faces = max(600, min(3500, len(mesh.faces) // 12))
        simple = mesh.simplify_quadric_decimation(face_count=faces)
        output = destination / filename
        simple.export(output)
        records.append({"file": filename, "original_triangles": len(mesh.faces), "visual_triangles": len(simple.faces), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (destination / "provenance.json").write_text(json.dumps({"description": "Derived from the Apache-2.0 Menagerie SO-101 models. Visual-only quadric decimation; collision meshes are unchanged.", "tool": "trimesh + fast-simplification", "files": records}, indent=2) + "\n", encoding="utf-8")
    print('Unique visual mesh triangles:', sum(r['original_triangles'] for r in records), '->', sum(r['visual_triangles'] for r in records))


if __name__ == '__main__':
    main()
