"""Post-processing utilities shared across experiment backends."""
import dill as pkl
from jax import vmap

# local
from _file_utils.make_paths import data_dir

def generate_trajsplines(
    model,
    states_stack: dict,
    ctrl_splines: dict,
    physics_params: dict,
    ctrl_keys: list,
    save_dir: str,
    ftype: str,
) -> dict:
    """Re-roll the executed control splines from each visited state.

    For every timestep of a completed experiment, the selected control spline
    is rolled out under the model dynamics for every parameter particle,
    producing the family of predicted trajectories (one per particle) that
    the planner reasoned over. The result is pickled for animation/plotting.

    Args:
        model: experiment interaction model exposing ``rollout(x, us, params)``.
        states_stack: logged states, one entry per timestep (per state key).
        ctrl_splines: logged control-spline stack, containing one executed
            spline per timestep for each control key, plus the ``'star'``
            selection indices.
        physics_params: logged parameter-particle stack, one entry per timestep.
        ctrl_keys: control keys of the experiment (e.g. ``['ee1', 'ee2']``).
        save_dir: experiment save directory (e.g. ``'dynamic_waiter'``).
        ftype: file name (without extension) for the pickled result.

    Returns:
        Dictionary of predicted trajectory splines stacked over time,
        particles, and horizon, with the ``'star'`` indices attached.
    """
    states_stack = {key: val[:-1] for key, val in states_stack.items()}
    us = {key: ctrl_splines[key] for key in ctrl_keys}
    traj_splines = vmap(
        lambda x, u, params: vmap(model.rollout, in_axes=(None, None, 0))(x, u, params)
    )(states_stack, us, physics_params)
    traj_splines['star'] = ctrl_splines['star']

    with open(f'{data_dir(save_dir)}/splines/{ftype}.pkl', 'wb') as file:
        pkl.dump(traj_splines, file)

    return traj_splines
