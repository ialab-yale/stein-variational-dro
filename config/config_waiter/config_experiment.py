import jax.numpy as jnp
from jax import vmap
import jax

from .sim import Dynamic_Waiter_Interaction_Model
from ..predictive_sampler import PredictiveSampler


class Experiment:
    """Dynamic waiter experiment backend.

    A table (robot, key ``'t'``) balances and transports a block (key
    ``'b'``) under uncertain block inertia, mass, and friction. Exposes the
    task Lagrangians (nominal, risk-sensitive, direct-DRO, DuSt) and the
    belief-sampling utilities consumed by the planning methods.
    """

    def __init__(self):
        self.save_dir = 'dynamic_waiter'
        self.num_param_samples = 5
        self.alpha = 0.999

        self.bounds = {
            'ctrl_bounds': ([-10.0, -0.2, -0.0], [10.0, 0.2, 0.0]),
            'pos_bounds': ([-10.0, -0.05, -0.0 * jnp.pi / 2.0], [10.0, 0.05, 0.0 * jnp.pi / 2.0]),
            'vel_bounds': ([-0.75, -0.05, -0.0], [0.75, 0.05, 0.0])
        }

        self.Qb = jnp.diag(jnp.hstack([20.0, 0.0, 10.0, 0.5, 0.01, 0.1]))
        self.Qbf = self.Qb * 1.0
        self.Qt = jnp.diag(jnp.hstack([0.0, 0.0, 10.0, 0.01, 0.01, 0.1]))
        self.Qtf = self.Qt * 1.0
        self.R = jnp.diag(jnp.hstack([0.0, 0.0, 0.0]))

        self.T = 1000
        self.model = Dynamic_Waiter_Interaction_Model(None, self.bounds)
        self.x0 = {'t': jnp.hstack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                   'b': jnp.hstack([0.3, (self.model.block_height + self.model.table_height) / 2.0 + 0.005, 0.0, 0.0, 0.0, 0.0])}
        self.model.true_states = self.x0

        self._lower_bnd = {'inertia': 1e-4, 'mass_block': 5e-2, 'mu': 1e-4}
        self._upper_bnd = {'inertia': 10.0, 'mass_block': 1.0, 'mu': 2.0}
        self.prior_keys = list(self._lower_bnd.keys())

        self.u0 = {'t': jnp.hstack([0.0, 0.0, 0.0])}
        self.ctrl_keys = list(self.u0.keys())
        self.ctrl_splines = {'t': [], 'star': [], 'splines': {'t': []}}

        self.ctrl_hrzn = 50
        self.num_splines = 50
        self.sampler = PredictiveSampler(
            dt=self.model.dt,
            ctrl_hrzn=self.ctrl_hrzn,
            ctrl_bounds=self.bounds['ctrl_bounds'],
            pos_bounds=self.bounds['pos_bounds'],
            vel_bounds=self.bounds['vel_bounds'],
            num_splines=self.num_splines,
            # waiter-specific control-bound adaptation with the running cost
            adjust_bounds_fn=lambda cost: (
                [-0.5 * cost**2.0 * 5.0, -0.2, -0.5],
                [0.5 * cost**2.0 * 5.0, 0.2, 0.5]
            )
        )

        def update_plan_and_keys(sample_nomplans, keys, star):
            # update the nominal plan and keys
            nominal_plan = self.sampler.update_plan(star, sample_nomplans, keys)
            keys = self.sampler.update_keys(star, keys)
            return nominal_plan, keys

        def get_ave_params(physics_params: dict):
            return {key: jnp.mean(val) for key, val in physics_params.items()}

        def get_block_goal_state(xt):
            return jnp.hstack([xt[0], 0.1, 0.0, 0.0, 0.0, 0.0])

        def init_random_key(seed):
            return jax.random.PRNGKey(seed)

        def update_samples(key, is_init=False):
            if not is_init: key = jax.random.split(key)[-1]
            return key, {_key: jax.random.uniform(
                key, shape=(self.num_param_samples,), minval=self._lower_bnd[_key], maxval=self._upper_bnd[_key]
            ) for _key in self.prior_keys}

        def bdry_constraints(p_t):
            left_bdry = lambda _p_t: 10.0 * jnp.minimum(0.0, _p_t + 0.2)**2.0
            right_bdry = lambda _p_t: 10.0 * jnp.minimum(0.0, 0.2 - _p_t)**2.0
            return jnp.sum(vmap(left_bdry)(p_t) + vmap(right_bdry)(p_t))

        # task lagrangian functions
        @jax.jit
        def lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, physics_params)
            p_t, v_t = jnp.split(tau['t'], 2, axis=1)
            p_b, v_b = jnp.split(tau['b'], 2, axis=1)
            block_goal = self.get_block_goal_state(tau['t'][0])

            return jnp.sum(vmap(lambda x_t, x_b, u: (x_b - block_goal).T @ self.Qb @ (x_b - block_goal) + x_t.T @ self.Qt @ x_t + u['t'].T @ self.R @ u['t'])(tau['t'], tau['b'], us)) \
                + bdry_constraints(p_t), tau

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
            L_DRO = jnp.max(_lambda * Dkl_epsilon + E_L + 1.0 / (2.0 * _lambda) * V_L)
            return L_DRO, tau

        @jax.jit
        def dust_lagrangian(x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, self.get_ave_params(physics_params))
            L_mc = vmap(self.lagrangian, in_axes=(None, None, 0))(x_init, us, physics_params)[0]
            return jnp.log(jnp.mean(jnp.exp(L_mc))), tau

        # MLE model
        def mle_cost(x_data: dict, x_init: dict, us: dict, physics_params: dict) -> float:
            tau = self.model.rollout(x_init, us, physics_params)
            total_cost = 0.0
            for key in tau:
                total_cost += jnp.sum((x_data[key] - tau[key][-1])**2.0)
            return total_cost

        def running_cost(states):
            return (states['b'] - self.get_block_goal_state(states['t'])).T @ self.Qb @ (states['b'] - self.get_block_goal_state(states['t'])) + bdry_constraints(states['t'][:2])

        def terminate(cost: float, fail_tol: float = 10.0, success_tol: float = 0.01) -> bool:
            if cost > fail_tol:
                # print('FAIL! Terminating experiment...')
                return True
            elif cost <= success_tol:
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
        self.get_block_goal_state = get_block_goal_state
