"""Rendering entry points for the SV-DRO experiments.

Each backend exposes ``render_<experiment>(idx, method, verbose)`` which reads
the saved trajectory at ``data/<save_dir>/pkl/traj/<method>_states-idx<idx>.pkl``
and writes an MP4 under ``videos/<save_dir>/mp4/mujoco/``.
"""


def render_experiment(experiment_name: str, idx: int, method: str, verbose: bool = True) -> str:
    """Render one trial for the selected experiment backend.

    Args:
        experiment_name: 'waiter' or 'bimanual'.
        idx: trial index of the saved trajectory.
        method: planning method name ('SVDRO', 'EMPPI', 'MPC', 'DRO').
        verbose: if False, suppress the render's progress printing.

    Returns:
        Path of the written MP4 file.
    """
    name = experiment_name.lower()
    if name == 'waiter':
        from .render_waiter import render_waiter
        return render_waiter(idx, method, verbose=verbose)
    elif name == 'bimanual':
        from .render_bimanual import render_bimanual
        return render_bimanual(idx, method, verbose=verbose)
    else:
        raise ValueError(f"unknown experiment backend: {experiment_name!r}")