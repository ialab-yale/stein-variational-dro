import jax
import jax.numpy as jnp
from jax import vmap


class PredictiveSampler:
    """Predictive-sampling action selection (https://arxiv.org/abs/2212.00541)."""

    def __init__(
            self,
            dt: float = 0.005,
            ctrl_hrzn: int = 10,
            ctrl_bounds: tuple = ([-1.0], [1.0]),
            pos_bounds: tuple = ([-0.5], [0.5]),
            vel_bounds: tuple = ([-0.5], [0.5]),
            num_splines: int = 10,
            adjust_bounds_fn=None,
    ):
        # default
        self.ctrl_bounds = ctrl_bounds
        self.num_splines = num_splines

        # optional experiment-specific control-bound adaptation (see adjust_ctrls)
        self.adjust_bounds_fn = adjust_bounds_fn

        # set default control limits
        u_min_default, u_max_default = jnp.hstack(ctrl_bounds[0]), jnp.hstack(ctrl_bounds[1])

        # set kinematic limits
        q_min, q_max = jnp.hstack(pos_bounds[0]), jnp.hstack(pos_bounds[1])
        qdot_min, qdot_max = jnp.hstack(vel_bounds[0]), jnp.hstack(vel_bounds[1])

        # sampling variance (keep at 1)
        var = 1.0

        # control horizon
        self.P = ctrl_hrzn

        def get_ctrl_limits(q: jnp.ndarray, qdot: jnp.ndarray) -> tuple:
            """
            returns new control limits given current system state
            """
            u_max = jnp.minimum(
                jnp.minimum(
                    u_max_default, (qdot_max - qdot) / dt
                ),
                (q_max - q) / dt
            )
            u_min = jnp.maximum(
                jnp.maximum(
                    u_min_default, (qdot_min - qdot) / dt
                ),
                (q_min - q) / dt
            )
            return u_min, u_max

        def get_init_nominal_plan(ctrl_keys: list) -> dict:
            """
            initializes the nominal plan for each control key
            """
            def get_plan_i():
                if len(self.ctrl_bounds[0]) == 1:
                    return jnp.zeros((self.P,))
                else:
                    return jnp.zeros((self.P, len(self.ctrl_bounds[0])))

            return {key: get_plan_i() for key in ctrl_keys}

        def get_init_keys(seed: int, ctrl_keys: list) -> dict:
            """
            initializes control sampling keys for each control key
            """
            def get_key_i(seed_i):
                return jax.random.split(jax.random.PRNGKey(seed_i), num=self.num_splines)

            return {key: get_key_i(seed + itr) for itr, key in enumerate(ctrl_keys)}

        self.get_init_plan = get_init_nominal_plan
        self.get_init_keys = get_init_keys

        def update_plan(star, sample_nomplans, keys):
            # update nominal_plan around the best candidate
            return {
                _key:
                val[star]
                + var**2 * jax.random.normal(keys[_key][star], shape=val[star].shape)
                for _key, val in sample_nomplans.items()
            }

        def update_keys(star, keys):
            # update control sampling keys
            return {
                _key:
                jax.random.split(val[star], num=self.num_splines)
                for _key, val in keys.items()
            }

        self.update_plan = update_plan
        self.update_keys = update_keys

        def cubic_hermite_spline(key, nominal_plan, x):
            """
            returns the control splines sampled from the nominal plan
            """
            # obtain nominal plan samples
            u_min, u_max = get_ctrl_limits(*jnp.split(x, 2))
            sample_nomplan = jnp.clip((var**2 * jax.random.normal(key, shape=nominal_plan.shape) + nominal_plan), u_min, u_max)

            # obtain weights
            w = jnp.linspace(0.0, 1.0, self.P)
            w_vec = jnp.vstack([jnp.ones_like(w), w, w**2, w**3]).T

            def compute_ctrl(w_vec_T_j, theta_jm1, theta_j, theta_jp1, theta_jp2):
                """
                computes control spline across all samples
                """
                # compute phis via finite difference method
                phi_j = 0.5 * ((theta_jp1 - theta_j) / dt + (theta_j - theta_jm1) / dt)
                phi_jp1 = 0.5 * ((theta_jp2 - theta_jp1) / dt + (theta_jp1 - theta_j) / dt)
                theta_vec = jnp.vstack([theta_j, phi_j, theta_jp1, phi_jp1])

                # scaling matrix
                scaling_mat = jnp.array([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, dt, 0.0, 0.0],
                                        [-3.0, -2.0 * dt, 3.0, -dt],
                                        [2.0, dt, -2.0, dt]])

                return w_vec_T_j @ scaling_mat @ theta_vec

            return jnp.clip(
                    vmap(
                    compute_ctrl,
                    in_axes=(0, 0, 0, 0, 0)
                )(
                    w_vec[1:-2],
                    sample_nomplan[:-3],
                    sample_nomplan[1:-2],
                    sample_nomplan[2:-1],
                    sample_nomplan[3:]
                ),
                u_min_default,
                u_max_default
            ), sample_nomplan

        def adjust_ctrls(cost):
            """optionally adapt the sampler's control bounds using the running cost"""
            if self.adjust_bounds_fn is not None:
                self.ctrl_bounds = self.adjust_bounds_fn(cost)

        def get_best_act(cost_fun, keys, nominal_plan, x_init, physics_params):
            """
            returns best control input based on cost
            """
            # get control splines for each control key
            control_splines = {}
            sample_nomplans = {}
            for key in keys:
                control_splines_i, sample_nomplans_i = vmap(
                    cubic_hermite_spline, in_axes=(0, None, None)
                )(keys[key], nominal_plan[key], x_init[key])
                control_splines[key] = control_splines_i
                sample_nomplans[key] = sample_nomplans_i

            # use splines to compute costs
            costs, trajsplines = vmap(cost_fun, in_axes=(None, 0, None))(x_init, control_splines, physics_params)
            star = jnp.argmin(costs)
            cost_min = jnp.min(costs)
            u = {key: val[star] for key, val in control_splines.items()}
            ustar = {key: val[star][0] for key, val in control_splines.items()}

            return {
                'u': u,
                'ustar': ustar,
                'star': star,
                'usplines': control_splines,
                'sample_nomplans': sample_nomplans,
                'trajsplines': trajsplines,
                'costs': costs,
                'cost_min': cost_min
            }

        self.get_action = get_best_act
        self.adjust_ctrls = adjust_ctrls
        self.spline = cubic_hermite_spline
