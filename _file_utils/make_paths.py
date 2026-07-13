import os

def data_dir(save_dir: str) -> str:
    """Root directory for pickled experiment data of a backend."""
    return f'data/{save_dir}/pkl'

def video_dir(save_dir: str) -> str:
    """Root directory for rendered videos of a backend."""
    return f'videos/{save_dir}/mp4'

def make_dirs(save_dir: str):
    """Create the output directory tree for an experiment backend.
    Data is saved under data/<save_dir>/pkl/... and videos under
    videos/<save_dir>/mp4.
    """
    pkl_dirs = ['losses', 'params', 'splines', 'traj', 'scene']
    [os.makedirs(f'{data_dir(save_dir)}/{dir}', exist_ok=True) for dir in pkl_dirs]
    os.makedirs(video_dir(save_dir), exist_ok=True)