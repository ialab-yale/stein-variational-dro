import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacfwd, jit, random
import dill as pkl
from tqdm import tqdm
import os

class Tangent_Jacobian_Lib:

    def __init__(self):

        def SVD_nullspace_basis(J):
            """
            returns nullspace orthonormal basis vectors of (normal) jacobian, J in R^3
            Args:
                `J`[jnp.ndarray]: normal jacobian in R^3 
            """
            # Build 3x1 row, do SVD to get nullspace basis (size 2)
            Jrow = J.reshape((1,3))
            # SVD on Jrow: U (1x1), S (1), Vt (3x3)
            _, S, Vt = jnp.linalg.svd(Jrow, full_matrices=True)
            # Nullspace (orthonormal) basis are the last two columns of V (rows of Vt)
            V = Vt.T
            N = V[:, 1:3]   # shape (3,2)

            return N
        
        def QR_nullspace_basis(Jn, eps=1e-2):
            """
            Jn: shape (3,)  -- normal Jacobian (row/gradient), can be jnp.array
            returns N: shape (3,2) with orthonormal columns spanning ker(Jn)
            """
            Jn = Jn.reshape((3,))           # ensure shape
            norm2 = jnp.dot(Jn, Jn)
            # If Jn is (almost) zero, return standard XY basis (no preferred normal)
            def compute_basis():
                P = jnp.eye(3) - jnp.outer(Jn, Jn) / norm2   # 3x3 projector
                A = jnp.array([[1., 0.],
                            [0., 1.],
                            [0., 0.]])                     # 3x2 seed
                B = P @ A                                     # 3x2
                # thin QR factorization: Q is 3x2, R is 2x2
                Q, R = jnp.linalg.qr(B, mode='reduced')
                # Ensure sign-consistent Q (optional): make R diagonal positive
                diag_sign = jnp.sign(jnp.diag(R))
                diag_sign = jnp.where(diag_sign == 0., 1., diag_sign)
                Q = Q * diag_sign[jnp.newaxis, :]
                return Q

            def fallback_basis():
                # Jn ~ 0: return standard (x,y) basis but orthonormal
                return jnp.array([[1., 0.],
                                [0., 1.],
                                [0., 0.]])

            N = jnp.where(norm2 > eps, compute_basis(), fallback_basis())
            return N
        
        def Jt_no_rotation(J, _):
            """
            returns tangent directional with no rotational contribution
            Args:
                `J`[jnp.ndarray]: normal jacobian in R^3 
            Returns:
                `t`[jnp.ndarray]: tangent direction vector
            """
            return jnp.hstack([-J[1], J[0], 0.0])
        
        def Jt_nullspace_method(J, v):
            """
            returns tangent directional based on inst. velocity
            Args:
                `J`[jnp.ndarray]: normal jacobian in R^3 
                `v`[jnp.ndarray]: velocity (body twist), in se(2) we have [vx, vy, vphi]
            Returns:
                `t`[jnp.ndarray]: tangent direction vector
            """
            # N = QR_nullspace_basis(J)
            N = SVD_nullspace_basis(J)
            # orthonormal columns so projection is N N^T u
            t = N @ (N.T @ v)
            return -t / (jnp.linalg.norm(t) + 1e-12)

        def Jt_min_inertia_method(J, M):
            """
            returns tangent directional based minimum kinetic energy law
            Args:
                `J`[jnp.ndarray]: normal jacobian in R^3 
                `M`[jnp.ndarray]: mass matrix in R^3x3
            """
            # M is 3x3 SPD inertia matrix
            N = QR_nullspace_basis(J)     
            K = N.T @ M @ N            
            # eigen-decompose K
            eigvals, eigvecs = jnp.linalg.eigh(K)
            # pick eigenvector for smallest eigenvalue
            a = eigvecs[:, 0]         
            t = N @ a                 
            t = t / (jnp.linalg.norm(t) + 1e-12)
            return t
        
        self.Jt_nullspace = Jt_nullspace_method
        self.Jt_mininert = Jt_min_inertia_method
        self.Jt_norot = Jt_no_rotation

class Dynamic_Waiter_Interaction_Model:

    def __init__(self, x0: dict[jnp.ndarray], bounds: dict, is_param_save: bool = True):
        # define global physics parameters
        self.true_states = x0                                                                               # system state, [m, m/s]
        self.dt = 0.005                                                                                     # timestep, sec, 0.005
        self.g = 9.81                                                                                       # gravity, m/s^2
        self.g_vec = jnp.hstack([0.0,self.g,0.0])                                                           # gravity vector
        self.mass_table = 0.50                                                                              # table mass, kg
        self.mass_block = 0.5                                                                              # mass of block, kg  
        self.table_height = 0.05                                                                            # table height, m (->y-dir from table frame)
        self.table_width = 1.00                                                                             # table width, m (->x-dir from table frame), 1.00
        self.block_height = 0.3                                                                            # block height, m (->y-dir from block frame)
        self.block_width = 0.1                                                                              # block height, m (->x-dir from block frame)
        self.n_per_edge = 10                                                                                # number of points per edge
        self.total_col_pts = self.n_per_edge * 4                                                            # total number of collision points
        self.inertia_table = 1.0/12.0 * self.mass_table * (self.table_height**2.0 + self.table_width**2.0)  # table inertia, kg m^2
        self.normal_contact_model_params = {
            'K': 75.0,                     # contact stiffness, N/m 100.0
            'C': 3.0                        # contact dampening, Ns/m # 1.0
        }
        self.friction_contact_model_params = {
            'mu': 0.3,                      # coefficient of friction
        }

        self.pos_bounds = jnp.array(bounds['pos_bounds'])
        self.vel_bounds = jnp.array(bounds['vel_bounds'])

        self.robot_sensor_variance = 1e-5
        self.block_tag_sensor_variance = 1e-3 # 5e-1

        self.tangent_jacobian_lib = Tangent_Jacobian_Lib()

        # save params
        if is_param_save:
            os.makedirs('data/dynamic_waiter/pkl/scene', exist_ok=True)
            with open('data/dynamic_waiter/pkl/scene/scene_data.pkl', 'wb') as file:
                    pkl.dump({
                        'table_width': self.table_width,
                        'table_height': self.table_height,
                        'block_width': self.block_width,
                        'block_height': self.block_height
                    }, file)

        def R(angle: float):
            """
            z-Rotation matrix
            """
            return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                            [jnp.sin(angle), jnp.cos(angle)]]).reshape((2,2))

        def get_inertia_block():
            """
            returns block inertia
            """
            I_centroid = 1.0/12.0 * self.mass_block * (self.block_width**2.0 + self.block_height**2.0)
            I_shifted = I_centroid # + self.mass_block*(y_shifted**2.0 + x_shifted**2.0)
            return I_shifted
        
        def get_block_vertices(qb: jnp.ndarray, physics_params: dict):
            """
            returns vertices (x4) of block
            """
            qbx, qby, qbphi = jnp.split(qb, 3)

            x1 = jnp.linspace(-self.block_width/2,  self.block_width/2, self.n_per_edge).flatten()
            y1 = jnp.full(self.n_per_edge, -self.block_height/2).flatten()

            x2 = jnp.full(self.n_per_edge,  self.block_width/2).flatten()
            y2 = jnp.linspace(-self.block_height/2,  self.block_height/2, self.n_per_edge).flatten()

            x3 = jnp.linspace( self.block_width/2, -self.block_width/2, self.n_per_edge).flatten()
            y3 = jnp.full(self.n_per_edge,  self.block_height/2).flatten()

            x4 = jnp.full(self.n_per_edge, -self.block_width/2).flatten()
            y4 = jnp.linspace( self.block_height/2, -self.block_height/2, self.n_per_edge).flatten()

            x = jnp.concatenate([x1, x2, x3, x4])
            y = jnp.concatenate([y1, y2, y3, y4])

            vertices_wrt_body = jnp.stack([x, y], axis=1)

            vertex_wrt_world = lambda vertex : jnp.hstack([qbx,qby]) + R(-qbphi)@vertex

            return vmap(vertex_wrt_world)(vertices_wrt_body)
        
        def phi_n(qt: jnp.ndarray, qb: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            """
            returns the signed distance function for point mass body with a box
            Args:
                `q`[jnp.ndarray]: pose of table
                `qb`[jnp.ndarray]: pose of block
            """
            def softmax(a, b, eps=0.001):
                """
                softmax activation
                """
                return 0.5 * (a + b + jnp.sqrt((a-b)**2 + eps**2))
            
            def phi_n_to_vertex(vertex):
                # split up the state vector of table and block
                qx, qy, qphi = jnp.split(qt, 3)
                # sdf vector
                phix, phiy = jnp.split(
                    jnp.abs(
                    R(-qphi)@(vertex - jnp.hstack([qx, qy]))
                    ) - jnp.hstack([0.5*self.table_width, 0.5*self.table_height]),
                    2
                )

                return (softmax(phix,0.0)**2 + softmax(phiy,0.0)**2)**0.5 \
                                -softmax(-softmax(phix,phiy),0.0)
                
            return vmap(phi_n_to_vertex)(get_block_vertices(qb, physics_params))
        
        def Jn_table_to_block(qt: jnp.ndarray, qb: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            jac =  jax.jacfwd(phi_n, argnums=0)(qt, qb, physics_params).reshape((self.total_col_pts,3))
            jac_normalized = vmap(lambda _jac : _jac / (jnp.linalg.norm(_jac)+1e-5))(jac)
            return jac_normalized
        
        def Jt_table_to_block(qt: jnp.ndarray, qb: jnp.ndarray, v_projected: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            Jt_i = lambda Jn : self.tangent_jacobian_lib.Jt_norot(Jn,v_projected)
            jac = vmap(Jt_i)(self.Jn_table_to_block(qt, qb, physics_params))
            jac_normalized = vmap(lambda _jac : _jac / (jnp.linalg.norm(_jac)+1e-5))(jac)
            return jac_normalized
        
        def Jn_block_to_table(qt: jnp.ndarray, qb: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            jac = jax.jacfwd(phi_n, argnums=1)(qt, qb, physics_params).reshape((self.total_col_pts,3))
            jac_normalized = vmap(lambda _jac : _jac / (jnp.linalg.norm(_jac)+1e-5))(jac)
            return jac_normalized
        
        def Jt_block_to_table(qt: jnp.ndarray, qb: jnp.ndarray, v_projected: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            Jt_i = lambda Jn : self.tangent_jacobian_lib.Jt_norot(Jn,v_projected)
            jac = vmap(Jt_i)(self.Jn_block_to_table(qt, qb, physics_params))
            jac_normalized = vmap(lambda _jac : _jac / (jnp.linalg.norm(_jac)+1e-5))(jac)
            return jac_normalized
        
        def get_normal_contact_force(_phi_n, _contact_vel, beta = 3.0):
            impulse_normal = lambda _phi_n, _contact_vel : 1.0/beta * jnp.log(
                1.0 + jnp.exp(beta*(-self.normal_contact_model_params['K']*_phi_n-self.normal_contact_model_params['C']*_contact_vel))
            )
            return jnp.clip(vmap(impulse_normal)(_phi_n, _contact_vel).flatten(), -100.0, 100.0)
        
        def get_friction_contact_force(normal_force, _contact_vel, physics_params, k = 500.0): 
            impulse_friction = lambda _normal_force, _contact_vel : -physics_params['mu']*_normal_force*jnp.tanh(k*_contact_vel)
            return vmap(impulse_friction)(normal_force, _contact_vel).flatten()
        
        def velocity_projected(jac: jnp.ndarray, v_total: jnp.ndarray):
            return vmap(lambda _jac : _jac@v_total)(jac)
        
        def force_projected(jac: jnp.ndarray, _lambda: jnp.ndarray):
            return jac.T@_lambda
        
        def get_forces(x: dict, physics_params: dict):
            # kinematic
            pt, vt  = jnp.split(x['t'], 2)
            pb, vb  = jnp.split(x['b'], 2)
            _phi_n = self.phi_n(pt, pb, physics_params)
            Jn_bt = self.Jn_block_to_table(pt, pb, physics_params)
            Jn_tb = self.Jn_table_to_block(pt, pb, physics_params)
            Jt_bt = self.Jt_block_to_table(pt, pb, vb - vt, physics_params)
            Jt_tb = self.Jt_table_to_block(pt, pb, vt - vb, physics_params)

            # dynamic 
            normal_force_bt = self.get_normal_contact_force(_phi_n, velocity_projected(Jn_bt, vb - vt))
            normal_force_tb = self.get_normal_contact_force(_phi_n, velocity_projected(Jn_tb, vt - vb))
            friction_force_bt = self.get_friction_contact_force(normal_force_bt, velocity_projected(Jt_bt, vb - vt), physics_params)
            friction_force_tb = self.get_friction_contact_force(normal_force_tb, velocity_projected(Jt_tb, vt - vb), physics_params)
            fn_bt_projected = force_projected(Jn_bt, normal_force_bt)
            fn_tb_projected = force_projected(Jn_tb, normal_force_tb)
            ft_bt_projected = force_projected(Jt_bt, friction_force_bt)
            ft_tb_projected = force_projected(Jt_tb, friction_force_tb)

            return {
                't': fn_tb_projected + ft_tb_projected,
                'b': fn_bt_projected + ft_bt_projected
            }
        
        def f(x : dict, u : jnp.ndarray, physics_params : dict) -> dict:
            # kinematic
            pt, vt  = jnp.split(x['t'], 2)
            pb, vb  = jnp.split(x['b'], 2)
            _phi_n = self.phi_n(pt, pb, physics_params)
            Jn_bt = self.Jn_block_to_table(pt, pb, physics_params)
            Jt_bt = self.Jt_block_to_table(pt, pb, vb - vt, physics_params)

            # dynamic 
            normal_force_bt = self.get_normal_contact_force(_phi_n, velocity_projected(Jn_bt, vb - vt))
            friction_force_bt = self.get_friction_contact_force(normal_force_bt, velocity_projected(Jt_bt, vb - vt), physics_params)
            fn_bt_projected = force_projected(Jn_bt, normal_force_bt)
            ft_bt_projected = force_projected(Jt_bt, friction_force_bt)

            mass_table = jnp.array([[self.mass_table,0.0,0.0],
                                    [0.0,self.mass_table,0.0],
                                    [0.0,0.0,self.inertia_table]])
            inv_mass_block = jnp.linalg.inv(jnp.array([[physics_params['mass_block'],0.0,0.0],
                                                    [0.0,physics_params['mass_block'],0.0],
                                                    [0.0,0.0,physics_params['inertia']]]))
            
            return {
                't': jnp.hstack(
                    [
                        jnp.clip(pt + self.dt * vt, self.pos_bounds[0], self.pos_bounds[1]).flatten(),
                        jnp.clip(vt + self.dt * jnp.linalg.inv(mass_table) @ u['t'], self.vel_bounds[0], self.vel_bounds[1]).flatten()
                    ]
                ),
                'b': jnp.hstack(
                    [
                        (pb + self.dt * vb).flatten(),
                        (vb + self.dt * (inv_mass_block @ (fn_bt_projected + ft_bt_projected) - self.g_vec)).flatten()
                    ]
                )
            }

        def measure(key, state: dict):
            _keys = jax.random.split(key, 2)
            return {'t': state['t'] + self.robot_sensor_variance**2.0 * jax.random.normal(_keys[0],shape=state['t'].shape),
                    'b': state['b'] + self.block_tag_sensor_variance**2.0 * jax.random.normal(_keys[1],shape=state['b'].shape)}
        
        
        @jax.jit
        def step(key: jnp.ndarray, x_true: dict, u : jnp.ndarray) -> dict:

            # TRUE PHYSICS PARAMETERS 
            true_physics_params = {
                                'inertia': self.get_inertia_block(),
                                'mass_block': self.mass_block,
                                'mu': self.friction_contact_model_params['mu'],
                                } 
            
            # obtain nest state dynamical information
            true_next_state = f(x_true, u, true_physics_params)
            next_state = measure(key, f(x_true,u,true_physics_params))
            next_forces = get_forces(x_true, true_physics_params)

            return true_next_state, next_state, next_forces

        def sim_fwd(key, ustar, running_cost):
            self.true_states, states, forces = self.step(key, self.true_states, ustar)
            cost = running_cost(self.true_states)
            return cost, states, forces
        
        @jax.jit
        def rollout(x_init : dict, us : dict, physics_params : dict):
            return jax.lax.scan(lambda x,u : (f(x,u,physics_params), x), init=x_init, xs=us, length=us['t'].shape[0])[-1]

        def animate_simulation(states_stack, trajsplines_stack, num_param_samples, experiment_string, idx):
            """Animate a completed experiment (requires the optional `animate` module)."""
            try:
                from .animate import animate
            except ImportError as e:
                raise ImportError(
                    'Animation requires the optional `animate` module in '
                    'config/config_waiter/. Install or add it to enable visualization.'
                ) from e
            return animate(states_stack, trajsplines_stack, num_param_samples, experiment_string, idx)

        self.phi_n = phi_n
        self.get_block_vertices = get_block_vertices
        self.get_inertia_block = get_inertia_block
        self.velocity_projected = velocity_projected
        self.force_projected = force_projected
        self.Jn_table_to_block = Jn_table_to_block
        self.Jn_block_to_table = Jn_block_to_table
        self.Jt_table_to_block = Jt_table_to_block
        self.Jt_block_to_table = Jt_block_to_table
        self.get_normal_contact_force = get_normal_contact_force
        self.get_friction_contact_force = get_friction_contact_force
        self.f = f
        self.step = step
        self.sim_fwd = sim_fwd
        self.rollout = rollout
        self.animate_simulation = animate_simulation