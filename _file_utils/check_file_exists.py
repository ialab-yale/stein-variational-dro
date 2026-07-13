import os

# local
from .make_paths import data_dir

def check_file_exists(idx: int, experiment_string: str, save_dir: str, enable: bool = True):
    if not enable:
        return False
    if os.path.exists(f'{data_dir(save_dir)}/losses/{experiment_string}_losses-idx{idx}.pkl'):
        return True
    return False