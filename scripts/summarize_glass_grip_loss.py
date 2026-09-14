"""Summarize preserved force replay; no simulator is instantiated."""
import contextlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put

folder=ROOT/'.run/glass-grip-loss-diagnosis-v1'
report_path=folder/'interpretation.json';log=folder/'interpretation.log'
if report_path.exists() or log.exists():raise FileExistsError('Preserve report and log.')
preflight=require_space(report_path,16*1024**2)
with log.open('x') as stream,contextlib.redirect_stdout(stream):
    put(folder/'source/scripts/summarize_glass_grip_loss.py',Path(__file__).read_bytes())
    rows=json.loads((folder/'contacts.json').read_text());model=json.loads((folder/'report.json').read_text())
    samples=[]
    for row in rows:
        entry={k:row[k] for k in ('time','tilt_deg','grip_q','grip_ctrl','grip_actuator_force','contact_wrench_world','glass_origin_in_tool')}
        for jaw in ('fixed','moving','external'):
            cs=[c for c in row['contacts'] if c['jaw']==jaw and c['normal_force']>.01]
            if not cs:entry[jaw]=dict(normal_n=0.);continue
            weights=np.array([c['normal_force'] for c in cs]);force=np.sum([c['force_world'] for c in cs],axis=0)
            normal=np.sum([np.array(c['normal_on_glass'])*c['normal_force'] for c in cs],axis=0)
            entry[jaw]=dict(normal_n=float(weights.sum()),height_mm=float(np.average([c['position_glass'][2] for c in cs],weights=weights)*1000),normal_vertical_n=float(normal[2]),friction_vertical_n=float(force[2]-normal[2]),net_vertical_n=float(force[2]),friction_cone_usage=float(np.average([c['cone_usage'] for c in cs],weights=weights)),relative_vertical_velocity_mm_s=float(np.average([c['relative_velocity_world'][2] for c in cs],weights=weights)*1000))
        samples.append(entry)
    times=(11.5,12.,12.5,13.,13.5,14.,14.2,14.3,14.4,14.5,14.55,14.575,14.6)
    selected=[min(samples,key=lambda x:abs(x['time']-t)) for t in times]
    text=[
      'The initial moving-jaw contact is already on the lip: normal-force-weighted height73.47mm at11.5s, compared with glass top74mm and commanded band69mm. The asymmetric pitched/hinged geometry places moving contact about4.47mm above the nominal band; fixed weighted contact is71.89mm.',
      'During early unsupported lift, the glass moves downward relative to both jaws at approximately0.5–1.3mm/s and gradually tilts. Strong contacts have substantial nominal friction margin (weighted cone usage about0.04–0.09), so this is not an initial Coulomb-limit shortfall. Finite soft-contact creep is a plausible contributor; that mechanism was not isolated by changing physics.',
      'Moving contacts remain at the top edge while the fixed jaw retains lower wall/lip contacts. At14.2s the moving weighted height is73.96mm and tilt7.76deg. At14.3s moving-jaw normal force has3.20N downward component, balanced by3.16N upward friction. This downward edge-normal component is nine times the glass weight; simply raising torque would also increase this wedging force.',
      'The glass then rotates/slides around the better-retained fixed side while the hinged moving jaw runs over the lip. Moving vertical slip reaches11.14mm/s at14.4s and13.60mm/s at14.5s; fixed-side vertical slip is only0.63 and0.87mm/s respectively. The final loss is asymmetric edge escape, not uniform downward translation of an intact two-pad pinch.',
      'The motor never intentionally releases: command tracks a closing offset and actual gripper actuator force remains−0.65Nm through the loss. Joint angle decreases0.4396→0.3981rad from11.5→14.575s; rapid closure follows lost contact.',
      'One targeted correction merits a feasibility check and, only if source and unchanged goal/route checks pass, one physical test: lower this same rigid grasp contact band to0.060m, keeping its exact local tool point/local body-up axis, opening0.85 and torque0.65Nm. First-order predicted moving contact height64.5mm leaves about9.5mm to the top, rather than0.5mm. The earlier3mm change was insufficient because actual moving contact still landed on the lip. This is a hypothesis, not a verified geometry/hold/placement; lower-band goal reachability must be checked rather than assumed.'
    ]
    report=dict(scope='Read-only interpretation of exact V17 motor replay; zero new manipulation trials or model edits.',preflight=preflight,replay_exact=model['replay_exact'],glass_weight_n=model['glass_mass']*9.81,contact_friction=[1.,1.,.005,.0005,.0005],findings=text,samples=selected,recommended_single_test=dict(contact_band_m=.060,local_tool_point=[.016844050389936402,-.0009278055682283223,-.0993204764644536],local_axis=[-.009721405847552073,.5015516051577634,.8650731076805114],torque_nm=.65,opening_rad=.85,required_precondition='Source, goal and rigid route geometry checks must pass; all physical guards and exact replay retained.'),limitations='One saved-run diagnosis. Contact softness mechanism not separately identified. Existing rendered views are occluded by the wrist; measured contact traces are the primary evidence.')
    put(report_path,report)
    try:
        import matplotlib
    except ModuleNotFoundError:
        print('Optional force plot unavailable: matplotlib is absent. Complete interpretation and contact traces were exported.')
        print(json.dumps(report,indent=2))
        sys.exit(0)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=[x for x in samples if x['time']>=11.5];t=[x['time'] for x in s]
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    for jaw,color in [('fixed','tab:blue'),('moving','tab:orange')]:
        axes[0].plot(t,[x[jaw].get('height_mm',np.nan) for x in s],label=jaw,color=color)
        axes[1].plot(t,[x[jaw]['normal_n'] for x in s],label=jaw,color=color)
    axes[0].axhline(74,color='black',linestyle='--',label='glass top');axes[0].set_ylabel('Contact height (mm)');axes[0].legend()
    axes[1].set_ylabel('Normal load (N)');axes[1].legend()
    axes[2].plot(t,[x['moving'].get('normal_vertical_n',np.nan) for x in s],label='moving normal Z')
    axes[2].plot(t,[x['moving'].get('friction_vertical_n',np.nan) for x in s],label='moving friction Z')
    axes[2].axhline(-report['glass_weight_n'],linestyle='--',color='black',label='gravity');axes[2].set_ylabel('Vertical force (N)');axes[2].legend()
    axes[3].plot(t,[x['tilt_deg'] for x in s],label='glass tilt');axes[3].set_ylabel('Tilt (degrees)');axes[3].set_xlabel('Continuation time (s)')
    ax=axes[3].twinx();ax.plot(t,[x['grip_q'] for x in s],color='tab:green',label='gripper angle');ax.set_ylabel('Gripper angle (rad)',color='tab:green')
    for axis in axes:axis.axvline(14.575,color='tab:red',linestyle=':');axis.grid(alpha=.2)
    fig.suptitle('V17 exact motor replay: moving jaw reaches and crosses the glass lip')
    path=folder/'force-sequence.png';require_space(path,4*1024**2)
    with path.open('xb') as imagefile:fig.savefig(imagefile,format='png',dpi=150)
    print(json.dumps(report,indent=2))
