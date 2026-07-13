import numpy as np
from tqdm import tqdm
import click

# local
from config.methods import make_methods
from render import render_experiment
from _file_utils.make_paths import make_dirs
from _file_utils.check_file_exists import check_file_exists

ENABLE_FILE_SKIPPING = False  # skip trials whose result files already exist


def main(method, outer_seed: int, experiment_name: str,
         render: bool = True, render_verbose: bool = False):
    """Run a single trial of a planning method, save it, then render it.

    Args:
        method: an instantiated planning method (SVDRO, EMPPI, MPC, or DRO).
        outer_seed: random seed for this trial.
        experiment_name: 'waiter' or 'bimanual', used to select the renderer.
        render: if True, render an MP4 of the trial right after saving.
        render_verbose: verbosity passed through to the render function.
    """
    if check_file_exists(method.idx, method.fname, method.save_dir, enable=ENABLE_FILE_SKIPPING):
        return

    _out = method.initialize(outer_seed)

    pbar = tqdm(range(method.T), desc='{}-{}'.format(method.fname, method.idx), leave=False)
    for i in pbar:
        _out = method.step(_out)
        pbar.set_postfix(cost=_out['cost'])
        if method.terminate(_out['cost']): break

    # print('cost: {}'.format(_out['cost']))

    # persist the trajectory (the renderer reads these pkl files from disk)
    method.save(_out['stacks'])

    # render the trial for the respective experiment
    if render:
        out_path = render_experiment(
            experiment_name,
            method.idx,
            method.fname,
            verbose=render_verbose
        )
        print('render -> {}'.format(out_path))


@click.command()
@click.option(
    '--experiment_name',
    type=click.Choice(['waiter', 'bimanual'], case_sensitive=False),
    default='waiter',
    help='Type of experiment backend.'
)
@click.option(
    '--num_trials',
    type=int,
    default=32,
    help='Number of trials per method.'
)
@click.option(
    '--render/--no-render',
    default=True,
    help='Render an MP4 after each trial (requires MuJoCo).'
)
@click.option(
    '--render_verbose',
    is_flag=True,
    default=False,
    help='Print render progress for each trial.'
)
def run(experiment_name: str, num_trials: int, render: bool, render_verbose: bool):
    SVDRO, EMPPI, MPC, DRO = make_methods(experiment_name)

    outer_seed = 0
    trial_pbar = tqdm(np.arange(num_trials), desc='trials')
    for idx in trial_pbar:
        for Method in (SVDRO, EMPPI, MPC, DRO):
            method = Method(idx)
            make_dirs(method.save_dir)
            main(method, outer_seed, experiment_name,
                 render=render, render_verbose=render_verbose)
        outer_seed += 1


if __name__ == "__main__":
    run()