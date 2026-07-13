import jax.numpy as jnp
from jax import vmap
import jax

from .sim import T_Block_Interaction_Model
from ..predictive_sampler import PredictiveSampler


class Experiment:
    """Bimanual T-block experiment backend.

    Two end effectors (keys ``'ee1'``, ``'ee2'``) push a planar T-shaped
    block (key ``'block'``) to a goal pose under uncertain block inertia,
    mass, and friction. Exposes the task Lagrangians (nominal,
    risk-sensitive, direct-DRO, DuSt) and the belief-sampling utilities
    consumed by the planning methods.
    """

    def __init__(self):
        self.save_dir = 'bimanual'
        self.num_param_samples = 5
        self.alpha = 0.999
        self.goal = jnp.hstack([0.2, 0.2, jnp.pi / 4.0, 0.0, 0.0, 0.0])

        self.bounds = {
            'ctrl_bounds': ([-1.0] * 2, [1.0] * 2),
            'pos_bounds': ([-1.0] * 2, [1.0] * 2),
            'vel_bounds': ([-1.0] * 2, [1.0] * 2)
        }

        self.Q = jnp.diag(jnp.hstack([5.0, 5.0, 2.0, 0.0, 0.0, 0.0]))
        self.Qf = self.Q * 1e-5
        self.R = jnp.eye(2) * 1e-5

        self.T = 1000
        self.model = T_Block_Interaction_Model(None, self.bounds, self.goal)
        self.x0 = {'ee1': jnp.hstack([0.5, 0.0, 0.0, 0.0]),
                   'ee2': jnp.hstack([-0.5, 0.0, 0.0, 0.0]),
                   'block': jnp.hstack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        self.model.true_states = self.x0

        self._lower_bnd = {'inertia': 1e-3, 'mass_block': 1e-3, 'mu': 1e-3}
        self._upper_bnd = {'inertia': 5.0, 'mass_block': 5.0, 'mu': 1.0}
        self.prior_keys = list(self._lower_bnd.keys())

        self.u0 = {'ee1': jnp.hstack([0.0, 0.0]),
                   'ee2': jnp.hstack([0.0, 0.0])}
        self.ctrl_keys = list(self.u0.keys())
        self.ctrl_splines = {'ee1': [], 'ee2': [], 'star': [], 'splines': {'ee1': [], 'ee2': []}}

        self.ctrl_hrzn = 50
        self.num_splines = 50
        self.sampler = PredictiveSampler(
            dt=self.model.dt,
            ctrl_hrzn=self.ctrl_hrzn,
            ctrl_bounds=self.bounds['ctrl_bounds'],
            pos_bounds=self.bounds['pos_bounds'],
            vel_bounds=self.bounds['vel_bounds'],
            num_splines=self.num_splines
        )

        def update_plan_and_keys(sample_nomplans, keys, star):
            # update the nominal plan and keys
            nominal_plan = self.sampler.update_plan(star, sample_nomplans, keys)
            keys = self.sampler.update_keys(star, keys)
            return nominal_plan, keys

        def get_ave_params(physics_params: dict):
            return {key: jnp.mean(val) for key, val in physics_params.items()}

        def init_random_key(seed):
            return jax.random.PRNGKey(seed)

        def update_samples(key, is_init=False):
            if not is_init: key = jax.random.split(key)[-1]
            return key, {_key: jax.random.uniform(
                key, shape=(self.num_param_samples,), minval=self._lower_bnd[_key], maxval=self._upper_bnd[_key]
            ) for _key in self.prior_keys}

        def angle_wrap(a):
            return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        def state_error(x_b, goal):
            return jnp.array([
                x_b[0] - goal[0],                  # position error
                x_b[1] - goal[1],                  # position error
                angle_wrap(x_b[2] - goal[2]),      # wrapped angle error
                x_b[3] - goal[3],                  # velocity error
                x_b[4] - goal[4],                  # velocity error
                x_b[5] - goal[5],                  # velocity error
            ])

        # task lagrangian functions
        @jax.jit
        def lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, physics_params)
            p_ee1, v_ee1 = jnp.split(tau['ee1'], 2, axis=1)
            p_ee2, v_ee2 = jnp.split(tau['ee2'], 2, axis=1)
            p_block, v_block = jnp.split(tau['block'], 2, axis=1)

            return jnp.sum(vmap(lambda x_block, u: state_error(x_block, self.goal).T @ self.Q @ state_error(x_block, self.goal)
                            + u['ee1'].T @ self.R @ u['ee1'] + u['ee2'].T @ self.R @ u['ee2'])(tau['block'], us)) \
                + state_error(tau['block'][-1], self.goal).T @ self.Qf @ state_error(tau['block'][-1], self.goal) \
                + jnp.sum(vmap(self.model.phi_n, in_axes=(0, 0, None))(p_ee1, p_block, physics_params)**2) \
                + jnp.sum(vmap(self.model.phi_n, in_axes=(0, 0, None))(p_ee2, p_block, physics_params)**2), tau

        def expectation_lagrangian(x_init: dict, us: dict, physics_params: dict):
            L_ave, _ = self.lagrangian(x_init, us, self.get_ave_params(physics_params))
            L_mc = jnp.mean(vmap(self.lagrangian, in_axes=(None, None, 0))(x_init, us, physics_params)[0])
            return L_mc - L_ave

        def variance_lagrangian(x_init: dict, us: dict, physics_params: dict):
            grad_L_ave, _ = jax.jacfwd(self.lagrangian, argnums=-1)(x_init, us, self.get_ave_params(physics_params))
            return jnp.sum(jnp.array([val**2.0 for val in grad_L_ave.values()]))

        @jax.jit
        def tilde_lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            E_L = self.expectation(x_init, us, physics_params)
            L_ave, tau = self.lagrangian(x_init, us, self.get_ave_params(physics_params))

            return L_ave + self.alpha * E_L, tau

        @jax.jit
        def direct_DRO_lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            _lambda = jnp.linspace(1e-2, 100.0, 1000)
            Dkl_epsilon = 0.001
            _, tau = self.lagrangian(x_init, us, self.get_ave_params(physics_params))
            E_L = self.expectation(x_init, us, physics_params)
            V_L = self.variance(x_init, us, physics_params)
            L_DRO = jnp.max(_lambda * Dkl_epsilon + _lambda * E_L + 1.0 / 2.0 * V_L)
            return L_DRO, tau

        @jax.jit
        def dust_lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, self.get_ave_params(physics_params))
            L_mc = vmap(self.lagrangian, in_axes=(None, None, 0))(x_init, us, physics_params)[0]
            return jnp.log(jnp.mean(jnp.exp(0.99 * L_mc))), tau

        # MLE model
        def mle_cost(x_data: dict, x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, physics_params)
            total_cost = 0.0
            for key in tau:
                total_cost += jnp.sum((x_data[key] - tau[key][-1])**2.0)
            return total_cost

        def running_cost(states):
            return state_error(states['block'], self.goal).T @ self.Q @ state_error(states['block'], self.goal)

        def terminate(cost: float) -> bool:
            if cost > 100.0:
                # print('FAIL! Terminating experiment...')
                return True
            elif cost <= 0.01:
                # print('SUCCESS! Terminating experiment...')
                return True
            return False

        self.init_random_key = init_random_key
        self.update_samples = update_samples
        self.lagrangian = lagrangian
        self.expectation = expectation_lagrangian
        self.variance = variance_lagrangian
        self.tilde_lagrangian = tilde_lagrangian
        self.direct_DRO_lagrangian = direct_DRO_lagrangian
        self.dust_lagrangian = dust_lagrangian
        self.mle_cost = mle_cost
        self.running_cost = running_cost
        self.terminate = terminate
        self.update = update_plan_and_keys
        self.get_ave_params = get_ave_params
