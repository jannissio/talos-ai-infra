"""A separately declared joint-scene candidate protocol, never a reachability oracle.

Every candidate uses the unchanged original seven-item sampler and ordinary
settling physics. We retain all candidates and select the first stable scene
without a separating certificate against BOTH arms for any item. Passing this
necessary initial-direct-contact screen does not certify a grasp or transfer.
Rejected scenes are not globally impossible: indirect rearrangement is outside
the screen. This protocol does not revise the original sampler or its results.
"""
import mujoco
import numpy as np

from .contact_reach import initial_direct_contact_separation
from .dinner import OBJECTS
from .random_dinner import assess, draw


SCHEMA = 'talos_joint_direct_contact_candidates_v1'


def contact_screen(model, data):
    rows = {name: {side: initial_direct_contact_separation(model, data, name, side)
                   for side in ('left', 'right')} for name in OBJECTS}
    separated = [name for name, arms in rows.items()
                 if all(row['excluded'] for row in arms.values())]
    return {'passes_necessary_screen': not separated, 'separated_items': separated,
            'objects': rows,
            'meaning': 'Initial direct contact only. A passing screen is unresolved manipulation feasibility; indirect rearrangement is not evaluated.'}


def draw_candidate(model, seed, *, record, before_candidate,
                   maximum_candidates=64, settling_steps=600):
    """Select a whole scene, preserving each reset through supplied callbacks.

    Callbacks receive no mutable live controller state. ``record(index, recipe,
    result, trace)`` must retain every candidate before the next is drawn.
    ``before_candidate(index)`` performs storage preflight before any new data.
    The returned MjData is a NEW reset-only scene, already settled. Subsequent
    manipulation must advance it by motor commands, with no state assignments.
    """
    if maximum_candidates < 1 or settling_steps < 1:
        raise ValueError('Candidate and settling budgets must be positive.')
    summaries = []
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for index in range(maximum_candidates):
        before_candidate(index)
        candidate_seed = seed + index
        data, recipe = draw(model, candidate_seed)
        initial = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, initial, state_spec)
        qpos, qvel, controls = [data.qpos.copy()], [data.qvel.copy()], []
        if recipe['generated']:
            for _ in range(settling_steps):
                assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                controls.append(data.ctrl.copy())
                mujoco.mj_step(model, data)
                qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy())
        mujoco.mj_forward(model, data)
        geometry = assess(model, data) if recipe['generated'] else None
        screen = contact_screen(model, data) if geometry and geometry['valid'] else None
        selected = bool(screen and screen['passes_necessary_screen'])
        reason = ('selected_unresolved_direct_contact_candidate' if selected else
                  'reset_generation_failed' if not recipe['generated'] else
                  'unstable_or_invalid_geometry' if not geometry['valid'] else
                  'at_least_one_initial_direct_contact_separation')
        trace = {'initial_integration': initial, 'qpos': np.asarray(qpos),
                 'qvel': np.asarray(qvel), 'ctrl': np.asarray(controls).reshape(-1, model.nu)}
        # Replay every ordinary settling step independently, including failures.
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, initial, state_spec); mujoco.mj_forward(model, replay)
        exact = True
        for tick, ctrl in enumerate(trace['ctrl']):
            replay.ctrl[:] = ctrl; mujoco.mj_step(model, replay)
            exact &= (np.array_equal(replay.qpos, trace['qpos'][tick+1])
                      and np.array_equal(replay.qvel, trace['qvel'][tick+1]))
        result = {'candidate_index': index, 'candidate_seed': candidate_seed,
                  'selected': selected, 'reason': reason, 'settling_frames': len(qpos),
                  'settling_motor_steps': len(controls), 'settling_replay_exact': bool(exact),
                  'geometry': geometry, 'direct_contact_screen': screen}
        record(index, recipe, result, trace)
        summaries.append({key: result[key] for key in ('candidate_index', 'candidate_seed',
            'selected', 'reason', 'settling_frames', 'settling_motor_steps', 'settling_replay_exact')})
        if not exact:
            raise RuntimeError('Candidate settling replay diverged; preserve this candidate.')
        if selected:
            recipe = dict(recipe, candidate_protocol=SCHEMA, protocol_seed=seed,
                          selected_candidate_index=index, already_settled_steps=settling_steps,
                          candidate_summaries=summaries, maximum_candidates=maximum_candidates,
                          manipulation_feasibility='unresolved',
                          indirect_rearrangement='not evaluated; rejected scenes remain outside this direct-start protocol only')
            return data, recipe
    return data, {'schema': SCHEMA, 'seed': seed, 'generated': False,
                  'maximum_candidates': maximum_candidates, 'candidate_summaries': summaries,
                  'reason': 'Declared candidate budget exhausted; all scenes retained.',
                  'manipulation_feasibility': 'No task-impossibility conclusion.'}
