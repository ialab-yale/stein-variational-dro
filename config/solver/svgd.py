import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class SVGD:
    """Stein Variational Gradient Descent over an ensemble of physics-parameter particles.

    Each call to ``evolve`` performs one SVGD transport step,

        theta_i <- theta_i + lr * phi(theta_i),

        phi(theta_i) = 1/M sum_j [ k(theta_j, theta_i) grad_{theta_j} log p(theta_j)
                                   + grad_{theta_j} k(theta_j, theta_i) ],

    Args:
        likelihood_model: 
            cost function from config_experiment
        _lower_bnd / _upper_bnd:
            Dictionaries of per-parameter box bounds used to clip particles.
        lr:
            Stein update rate.
        kernel_type:
            One of ``'rbf'`` (median-heuristic bandwidth), ``'imq'`` (inverse
            multiquadric), or ``'1'`` (constant kernel; repulsion vanishes and
            the update reduces to parallel gradient ascent).
        prior_type:
            One of ``'uniform'`` (improper flat prior, contributes zero) or
            ``'gaussian'`` (unit-variance Gaussian centered at the current
            ensemble mean).
    """

    def __init__(
        self,
        likelihood_model,
        _lower_bnd: dict,
        _upper_bnd: dict,
        lr: float = 1e-4,
        kernel_type: str = 'rbf',
        prior_type: str = 'uniform',
        bandwidth: float | None = None,
        min_bandwidth: float = 1e-3,
        imq_bandwidth: float = 0.5,
        imq_decay: float = 0.5,
    ):
        self.lr = lr
        self.likelihood_model = likelihood_model
        self._lower_bnd = _lower_bnd
        self._upper_bnd = _upper_bnd
        self.prior_keys = tuple(_lower_bnd.keys())
        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth

        _lower_vec = jnp.hstack([_lower_bnd[k] for k in self.prior_keys])
        _upper_vec = jnp.hstack([_upper_bnd[k] for k in self.prior_keys])

        # dict of arrays (M,) per key <-> matrix (M, D)
        def pack(theta: dict) -> jnp.ndarray:
            return jnp.stack([theta[k] for k in self.prior_keys], axis=1)

        def unpack_matrix(Theta: jnp.ndarray) -> dict:
            return {k: Theta[:, i] for i, k in enumerate(self.prior_keys)}

        def unpack_single(theta_vec: jnp.ndarray) -> dict:
            return {k: theta_vec[i] for i, k in enumerate(self.prior_keys)}

        # log priors (evaluated on a single packed particle vector)
        def log_prior_gaussian(theta_vec: jnp.ndarray, theta_star_vec: jnp.ndarray):
            return -0.5 * jnp.sum((theta_vec - theta_star_vec) ** 2.0)

        def log_prior_uniform(theta_vec: jnp.ndarray, theta_star_vec: jnp.ndarray):
            return 0.0

        # log posteriors (evaluated on a single packed particle vector)
        def adv_log_posterior(x_init: dict, us: dict, theta_star_vec, theta_vec):
            # adversarial target: ascend the task cost (worst-case parameters)
            theta = unpack_single(theta_vec)
            return self.log_prior(theta_vec, theta_star_vec) \
                + self.likelihood_model(x_init, us, theta)[0]

        # kernels (on the joint parameter vector)
        def median_heuristic_bandwidth(Theta: jnp.ndarray):
            if self.bandwidth is not None:
                return jnp.asarray(self.bandwidth)
            diffs = Theta[:, None, :] - Theta[None, :, :]
            sq_dists = jnp.sum(diffs ** 2.0, axis=-1)
            M = Theta.shape[0]
            h = jnp.median(sq_dists.reshape(-1)) / jnp.log(M + 1.0)
            return jnp.maximum(h, self.min_bandwidth)

        def kernel_rbf(theta_j, theta_i, h):
            return jnp.exp(-jnp.sum((theta_j - theta_i) ** 2.0) / h)

        def kernel_1(theta_j, theta_i, h):
            return 1.0

        def kernel_imq(theta_j, theta_i, h):
            return (imq_bandwidth + jnp.sum((theta_j - theta_i) ** 2.0)) ** (-imq_decay)

        # single SVGD transport step over the packed particle matrix
        def svgd_step(Theta: jnp.ndarray, log_posterior_single) -> jnp.ndarray:
            M = Theta.shape[0]
            h = median_heuristic_bandwidth(Theta)
            theta_star_vec = jnp.mean(Theta, axis=0)

            # score[j] = grad_{theta_j} log p(theta_j), shape (M, D)
            score = jax.vmap(
                jax.grad(lambda theta_vec: log_posterior_single(theta_star_vec, theta_vec))
            )(Theta)

            # K[j, i] = k(theta_j, theta_i), shape (M, M)
            K = jax.vmap(
                lambda theta_j: jax.vmap(
                    lambda theta_i: self.kernel(theta_j, theta_i, h)
                )(Theta)
            )(Theta)

            # attraction: sum_j k(theta_j, theta_i) score[j]
            attraction = K.T @ score  # (M, D)

            # repulsion: sum_j grad_{theta_j} k(theta_j, theta_i)
            def repulsion_for_target(theta_i):
                grad_k_wrt_source = jax.vmap(
                    lambda theta_j: jax.jacfwd(
                        lambda source: self.kernel(source, theta_i, h)
                    )(theta_j)
                )(Theta)  # (M, D)
                return jnp.sum(grad_k_wrt_source, axis=0)  # (D,)

            repulsion = jax.vmap(repulsion_for_target)(Theta)  # (M, D)

            phi = (attraction + repulsion) / M
            Theta_next = Theta + self.lr * phi
            return jnp.clip(Theta_next, _lower_vec, _upper_vec)

        # svgd updates (exposed)
        @jax.jit
        def evolve(x_init: dict, us: dict, theta: dict) -> dict:
            def logp(theta_star_vec, theta_vec):
                return adv_log_posterior(x_init, us, theta_star_vec, theta_vec)
            return unpack_matrix(svgd_step(pack(theta), logp))

        # choice of kernel
        if kernel_type == '1':
            self.kernel = kernel_1
        elif kernel_type == 'imq':
            self.kernel = kernel_imq
        elif kernel_type == 'rbf':
            self.kernel = kernel_rbf
        else:
            raise Exception('not a valid kernel type')

        # choice of prior
        if prior_type == 'uniform':
            self.log_prior = log_prior_uniform
        elif prior_type == 'gaussian':
            self.log_prior = log_prior_gaussian
        else:
            raise Exception('not a valid log prior type')

        # svgd step
        self.evolve = evolve
