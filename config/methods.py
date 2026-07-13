import jax
import jax.numpy as jnp
import numpy as np
import time
from importlib import import_module

from .logger import Logger
from .trajsplines import generate_trajsplines
from .solver.svgd import SVGD

TO_VISUALIZE = False        # enable/disable animation on save
NUM_CONTROL_KNOTS = 10      # executed knots per replanning cycle


def load_experiment(config_name: str):
    """Load Experiment class from selected config subpackage."""
    module = import_module(
        f".config_{config_name}.config_experiment",
        package=__package__
    )
    return module.Experiment


def make_methods(config_name: str = "waiter"):
    """Create method classes for a selected experiment backend.

    Returns the four planners:
        SVDRO:  Stein variational DRO. Plans with the risk-aware
                cost over the particle ensemble and transports
                the particles toward adversarial parameters via SVGD.
        EMPPI:  Ensemble MPPI-style baseline. Plans with the same ensemble
                objective but resamples particles via prior.
        MPC:    Point-estimate baseline. Collapses the belief to its mean and
                plans with the nominal Lagrangian.
        DRO:    Conventional DRO baseline. Plans with the dual (direct) DRO
                cost and resamples particles from the prior.
    """
    Experiment = load_experiment(config_name)

    class Exp_Utils(Experiment):
        """Shared initialize/step/save machinery for all methods."""

        def __init__(self, idx: int = 0):
            super().__init__()
            self.config_name = config_name
            self.idx = idx
            self.logger = Logger(self.save_dir)
            self.num_control_knots = NUM_CONTROL_KNOTS

        def initialize(self, outer_seed: int) -> dict:
            nominal_plan = self.sampler.get_init_plan(self.ctrl_keys)
            pred_sam_keys = self.sampler.get_init_keys(200, self.ctrl_keys)
            s_key = jax.random.PRNGKey(outer_seed)
            states = self.x0
            stacks = self.logger.init_experiment_loggers(self.x0, self.u0, self.ctrl_splines, self.prior_keys)
            _, physics_params_curr = self.update_samples(s_key, is_init=True)
            return {
                'nominal_plan': nominal_plan,
                'pred_sam_keys': pred_sam_keys,
                'states': states,
                'stacks': stacks,
                's_key': s_key,
                'physics_params_curr': physics_params_curr
            }

        def step(self, carry: dict) -> dict:
            # plan: select the best control spline under this method's cost
            _output = self.sampler.get_action(
                self.cost,
                carry['pred_sam_keys'],
                carry['nominal_plan'],
                carry['states'],
                carry['physics_params_curr']
            )
            nominal_plan, pred_sam_keys = self.update(
                _output['sample_nomplans'],
                carry['pred_sam_keys'],
                _output['star']
            )

            # execute: run the selected spline on the simulator
            stacks = carry['stacks']
            for k in range(self.num_control_knots):
                cost, states, forces = self.model.sim_fwd(
                    carry['s_key'],
                    {key: val[k] for key, val in _output['u'].items()},
                    self.running_cost
                )
                stacks = self.logger.log(self.model, stacks, _output, cost, carry['physics_params_curr'])

            # update: refresh the parameter belief (method-specific)
            s_key, physics_params_curr = self.update_belief(carry, states, _output)
            self.sampler.adjust_ctrls(cost)

            return {
                'cost': cost,
                'states': states,
                'stacks': stacks,
                's_key': s_key,
                'physics_params_curr': physics_params_curr,
                'nominal_plan': nominal_plan,
                'pred_sam_keys': pred_sam_keys,
            }

        def update_belief(self, carry, states, _output):
            """Method-specific parameter-belief update. Overridden by subclasses."""
            raise NotImplementedError

        def save(self, stacks: dict) -> None:
            stacks = self.logger.convert_to_numpy(stacks)
            # print('generating trajectory splines from controls...')
            trajsplines_stack = generate_trajsplines(
                self.model,
                stacks['states'],
                stacks['ctrlsplines'],
                stacks['physics_params'],
                ctrl_keys=self.ctrl_keys,
                save_dir=self.save_dir,
                ftype=f'{self.fname}_trajsplines-idx{self.idx}'
            )
            # print('done!')
            self.logger.save(stacks, self.idx, fname=self.fname)
            if TO_VISUALIZE:
                self.model.animate_simulation(
                    stacks['states'],
                    trajsplines_stack,
                    self.num_param_samples,
                    experiment_string=self.fname,
                    idx=self.idx
                )

    class SVDRO(Exp_Utils):
        """Stein variational distributionally robust planning."""

        def __init__(self, idx: int = 0, kernel_type: str = 'rbf'):
            super().__init__(idx)
            self.fname = 'SVDRO'
            self.kernel_type = kernel_type
            self.cost = self.tilde_lagrangian
            self.svgd = SVGD(self.lagrangian, self._lower_bnd, self._upper_bnd, kernel_type=kernel_type)

        def update_belief(self, carry, states, _output):
            s_key, _ = self.update_samples(carry['s_key'])
            physics_params_curr = self.svgd.evolve(states, _output['u'], carry['physics_params_curr'])
            return s_key, physics_params_curr

    class EMPPI(Exp_Utils):
        """Ensemble baseline: plans over the ensemble, resamples from the prior."""

        def __init__(self, idx: int = 0):
            super().__init__(idx)
            self.fname = 'EMPPI'
            self.cost = self.tilde_lagrangian

        def update_belief(self, carry, states, _output):
            return self.update_samples(carry['s_key'])

    class MPC(Exp_Utils):
        """Point-estimate baseline: plans around the (fixed) mean parameters."""

        def __init__(self, idx: int = 0):
            super().__init__(idx)
            self.fname = 'MPC'
            self.cost = self.lagrangian

        def initialize(self, outer_seed: int) -> dict:
            carry = super().initialize(outer_seed)
            carry['physics_params_curr'] = self.get_ave_params(carry['physics_params_curr'])
            return carry

        def update_belief(self, carry, states, _output):
            s_key, _ = self.update_samples(carry['s_key'])
            return s_key, carry['physics_params_curr']

        def save(self, stacks: dict) -> None:
            # point-estimate methods carry no particle ensemble,
            # no per-particle trajectory splines are generated
            stacks = self.logger.convert_to_numpy(stacks)
            self.logger.save(stacks, self.idx, fname=self.fname)
            if TO_VISUALIZE:
                self.model.animate_simulation(
                    stacks['states'],
                    None,
                    self.num_param_samples,
                    experiment_string=self.fname,
                    idx=self.idx
                )

    class DRO(Exp_Utils):
        """Direct DRO baseline: plans with the dual DRO objective, resamples from the prior."""

        def __init__(self, idx: int = 0):
            super().__init__(idx)
            self.fname = 'DRO'
            self.cost = self.direct_DRO_lagrangian

        def update_belief(self, carry, states, _output):
            return self.update_samples(carry['s_key'])

    return SVDRO, EMPPI, MPC, DRO
