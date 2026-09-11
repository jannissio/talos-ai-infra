"""Regenerate portable MJCF dinner objects from the canonical Python builders."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco

from simulation_lab.dinner import OBJECTS, add_drawer, make_object
from simulation_lab.scene import build_scene

OUTPUT = Path(__file__).resolve().parents[1]/"simulation_lab/assets/dinner"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ET.fromstring(build_scene(scenario="dinner")[0])
    materials = deepcopy(source.find("asset"))
    for mesh in list(materials.findall("mesh")):
        materials.remove(mesh)
    records = []
    for name in [*OBJECTS, "drawer"]:
        root = ET.Element("mujoco", model="talos_"+name)
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(root, "option", timestep=".005", integrator="implicitfast", cone="elliptic", iterations="30")
        root.append(deepcopy(materials))
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -.3 2", dir="0 0 -1")
        floor_z = .76 if name == "drawer" else 0.
        ET.SubElement(world, "geom", name="inspection_floor", type="plane", size=".4 .4 .05", pos=f"0 0 {floor_z}", rgba=".27 .32 .36 1")
        if name == "drawer":
            spec = add_drawer(world, False)
        else:
            spec = make_object(world, name, (0, 0, .001))
        ET.indent(root, space="  ")
        text = ET.tostring(root, encoding="unicode")+"\n"
        mujoco.MjModel.from_xml_string(text)
        path = OUTPUT/(name+".xml")
        path.write_text(text, encoding="utf-8", newline="\n")
        records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "spec": spec})
    (OUTPUT/"manifest.json").write_text(json.dumps({"schema": "talos.dinner-assets.v1", "license": "MIT",
        "source": "simulation_lab/dinner.py", "generator": "scripts/export_dinner_assets.py",
        "units": {"length": "m", "mass": "kg", "angle": "rad"}, "assets": records}, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"Exported and compiled {len(records)} standalone MJCF assets in {OUTPUT}")


if __name__ == "__main__":
    main()
