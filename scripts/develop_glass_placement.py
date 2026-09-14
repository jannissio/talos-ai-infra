"""Bounded upright-glass diagnostic from an actual exposed joint-scene state.

Uses the immutable recorder and configurable teacher in the adjacent glass
diagnostic; each run snapshots both harnesses and the exact imported teacher.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.develop_glass_reorientation import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--teacher-source')
    parser.add_argument('--initial-from',default='.run/whole-table-development-v18-seed4004/lookahead-01-034-00-glass')
    parser.add_argument('--replay-prefix',action='store_true')
    parser.add_argument('--grasp',default='vessel_rim_0.00000')
    parser.add_argument('--torque',type=float,default=.65)
    parser.add_argument('--tool-x',type=float)
    parser.add_argument('--tool-z',type=float)
    parser.add_argument('--rim-angle',type=float)
    parser.add_argument('--side-wrap',action='store_true')
    parser.add_argument('--refined-family',action='store_true')
    parser.add_argument('--pitch',type=float)
    parser.add_argument('--goal-contact',action='store_true')
    parser.add_argument('--goal-contact-height',type=float,default=.072)
    parser.add_argument('--require-full-preflight',action='store_true')
    parser.add_argument('--rim-radius',type=float,default=.022)
    parser.add_argument('--rim-height',type=float,default=.067)
    parser.add_argument('--closing-sign',type=int,choices=(-1,1),default=1)
    parser.add_argument('--target',type=float,nargs=3,default=[.14425000014816797,-.020749999937376695,.76])
    parser.add_argument('--arm',choices=('left','right','auto'),default='right')
    parser.add_argument('--branches',type=int,default=24)
    args=parser.parse_args()
    if not 0<args.torque<=2.94:parser.error('Torque must remain within real rated motors.')
    if (args.rim_angle is not None or args.side_wrap or args.pitch is not None) and (args.tool_x is None or args.tool_z is None):
        parser.error('Explicit rim candidates require both tool point coordinates.')
    for name,value in dict(seed=2026114004,mode='upright',side_axis=False,bearing=0.,post_hold_sideways=False,
            horizontal_plane=False,roll_bias=0.,align_duration=4.,carry_torque=None,carry_clearance=0.,reorient_corrections=0).items():
        setattr(args,name,value)
    run(args)
