"""Index verified completed teacher episodes for a shared spatial policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation_lab.spatial_demonstrations import build_inventory
from simulation_lab.storage import require_space


def main(output: Path, runs: list[Path], instruction_template: str,
         excluded_runs: list[Path] = (), include_independent_lookahead: bool = False,
         temporary_instruction_template: str =
         "move the {item} to temporary table cell row {row} column {column}"):
    if output.exists():
        raise FileExistsError("Preserve the existing spatial demonstration inventory.")
    inventory = build_inventory(runs, instruction_template=instruction_template,
        excluded_runs=excluded_runs, include_independent_lookahead=include_independent_lookahead,
        temporary_instruction_template=temporary_instruction_template)
    payload = (json.dumps(inventory, indent=2) + "\n").encode()
    preflight = require_space(output, len(payload) + 1024 * 1024)
    inventory["export_preflight"] = preflight
    payload = (json.dumps(inventory, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({key: inventory[key] for key in
          ("sample_count", "sample_count_by_item", "independent_scene_group_count",
           "sample_count_by_workflow_origin", "unique_exact_content_sample_count",
           "exclusion_count", "training_readiness")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path,
                        help="Explicit immutable completed run folder; repeat for inventory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-unfinished-run", action="append", default=[], type=Path,
                        help="Record a run as excluded at this inventory cutoff without reading it.")
    parser.add_argument("--instruction-template", default="place the {item}")
    parser.add_argument("--include-independent-lookahead", action="store_true",
                        help="Include replay-exact successful copied-state primitives with explicit origin labels.")
    parser.add_argument("--temporary-instruction-template",
                        default="move the {item} to temporary table cell row {row} column {column}")
    args = parser.parse_args()
    main(args.output, args.run, args.instruction_template, args.exclude_unfinished_run,
         args.include_independent_lookahead, args.temporary_instruction_template)
