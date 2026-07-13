import jax.numpy as jnp
import dill as pkl

# local
from _file_utils.make_paths import data_dir

class Logger:
    """Experiment data logger."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.data_dir = data_dir(save_dir)

        # initialize stacks
        def init_experiment_loggers(x0: dict, u0: dict, ctrl_splines0: dict, physics_params_keys: list):
            states_stack = {key: [val] for key, val in x0.items()}
            ctrls_stack = {key: [val] for key, val in u0.items()}
            ctrlssplines_stack = ctrl_splines0
            running_losses_stack = {'losses': []}
            physics_params_stack = {key: [] for key in physics_params_keys}
            return {
                'states': states_stack,
                'ctrls': ctrls_stack,
                'ctrlsplines': ctrlssplines_stack,
                'losses': running_losses_stack,
                'physics_params': physics_params_stack
            }

        # loggers
        def log_states_data(stack: dict, data: dict) -> dict:
            for key in stack:
                stack[key].append(data[key])
            return stack

        def log_ctrls_data(stack: dict, data: dict) -> dict:
            for key in stack:
                stack[key].append(data[key])
            return stack

        def log_splines_data(stack: dict, u_data: dict, usplines_data: dict, star_data: float) -> dict:
            for key in stack['splines']:
                stack[key].append(u_data[key])
                stack['splines'][key].append(usplines_data[key])
            stack['star'].append(star_data)
            return stack

        def log_losses_data(stack: dict, data: float) -> dict:
            stack['losses'].append(data)
            return stack

        def log_params_data(stack: dict, data: dict) -> dict:
            for key in stack.keys():
                stack[key].append(data[key])
            return stack

        def log_experiment_data(model, stacks: dict, _output: dict, cost: float, physics_params: dict):
            stacks['states'] = log_states_data(stacks['states'], model.true_states)
            stacks['ctrls'] = log_ctrls_data(stacks['ctrls'], _output['ustar'])
            stacks['ctrlsplines'] = log_splines_data(stacks['ctrlsplines'], _output['u'], _output['usplines'], _output['star'])
            stacks['losses'] = log_losses_data(stacks['losses'], cost)
            stacks['physics_params'] = log_params_data(stacks['physics_params'], physics_params)
            return stacks

        def convert_to_numpy(stacks: dict) -> dict:
            _convert = lambda stack: {key: jnp.array(val) if not isinstance(val, dict) else _convert(val) for key, val in stack.items()}
            return {key: _convert(val) for key, val in stacks.items()}

        def savetraj(stack: dict, ftype: str = 'optimizedtraj') -> None:
            with open(f'{self.data_dir}/traj/{ftype}.pkl', 'wb') as pickle_file:
                pkl.dump(stack, pickle_file)

        def savesplines(stack: dict, ftype: str = 'optimizedspline') -> None:
            with open(f'{self.data_dir}/splines/{ftype}.pkl', 'wb') as pickle_file:
                pkl.dump(stack, pickle_file)

        def savelosses(stack, ftype: str = 'optimizedlosses') -> None:
            with open(f'{self.data_dir}/losses/{ftype}.pkl', 'wb') as pickle_file:
                pkl.dump(stack, pickle_file)

        def saveparams(stack, ftype: str = 'optimizedparams') -> None:
            with open(f'{self.data_dir}/params/{ftype}.pkl', 'wb') as pickle_file:
                pkl.dump(stack, pickle_file)

        def save(stacks, idx, fname):
            _save_fns = {'states': savetraj, 'ctrls': savetraj, 'ctrlsplines': savesplines, 'losses': savelosses, 'physics_params': saveparams}
            _save_fnames = {_key: f'{fname}_{_key}-idx{idx}' for _key in _save_fns.keys()}
            {_key: _val(stacks[_key], _save_fnames[_key]) for _key, _val in _save_fns.items()}

        self.init_experiment_loggers = init_experiment_loggers
        self.log = log_experiment_data
        self.convert_to_numpy = convert_to_numpy
        self.save = save
