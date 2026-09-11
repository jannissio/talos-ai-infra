"""Rebuild synchronized images from a complete saved trajectory, without rerunning the robot."""
import argparse
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation_lab.recording import export_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.folder/"manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("trajectory_complete"):
        parser.error("The trajectory is incomplete; refusing to export it as a demonstration.")
    export_images(str(args.folder.resolve()), multiprocessing.get_context("spawn").Event())
    result = json.loads((args.folder/"manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(result["images"], indent=2))
    if result["images"]["status"] != "completed":
        raise SystemExit(1)
