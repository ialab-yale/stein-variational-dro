import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacfwd, jit, random
import dill as pkl
from tqdm import tqdm
from collections.abc import Callable

class T_Block_Interaction_Model:

    def __init__(self, x0: dict, bounds: dict, goal: jnp.ndarray, is_param_save: bool = True):
        # define global physics parameters
        self.true_states = x0               # system state, [m, m/s]
        self.dt = 0.005                     # timestep, sec
        self.g = 9.81                       # gravity, m/s^2
        self.mass_end_eff = 0.5             # end effector mass, kg 0.5
        self.mass_block = 0.5               # mass of block, kg
        self.T_height = 0.5                 # T height, m
        self.T_thickness = 0.2              # T height, m
        self.T_width = 0.6                  # T height, m
        self.r_end_eff = 0.05               # end effector radius, m
        self.contact_model_params = {
            'K': 100.0,                     # contact stiffness, N/m
            'C': 0.0                        # contact dampening, Ns/m
        }
        self.ground_contact_model_params = {
            'mu': 0.4,                      # coefficient of friction 0.8
        }

        self.pos_bounds = jnp.array(bounds['pos_bounds'])
        self.vel_bounds = jnp.array(bounds['vel_bounds'])

        self.robot_sensor_variance = 1e-4
        self.block_tag_sensor_variance = 1e-4

        self._goal = goal

        # self.policy_net = None
        # self.traj_length = None

        # save params
        if is_param_save:
            os.makedirs('data/bimanual/pkl/scene', exist_ok=True)
            with open('data/bimanual/pkl/scene/scene_data.pkl', 'wb') as file:
                    pkl.dump({
                        'T_height': self.T_height,
                        'T_width': self.T_width,
                        'T_thickness': self.T_thickness,
                        'r_end_eff': self.r_end_eff,
                        'goal': self._goal
                    }, file)

        
        def get_inertia_T():
            """
            returns the mass moment of intertia wrt body frame of object
            """
            # area of each rectangle and tota area
            A_top = self.T_thickness*self.T_width
            A_stem = self.T_thickness*self.T_height
            A_total = A_top + A_stem

            # compute fractional mass of each rectangle
            mass_top = self.mass_block * A_top / A_total
            mass_stem = self.mass_block * A_stem / A_total

            # compute rectangle inertias and total inertia
            I_top = (1.0 / 12.0) * mass_top * (self.T_width**2 + self.T_thickness**2)
            I_stem = (1.0 / 12.0) * mass_stem * (self.T_height**2 + self.T_thickness**2)
            I_total = I_top \
                    + I_stem \
                    + mass_stem *(self.T_height/4.0)**2 \
                    + mass_top *(self.T_height/4.0)**2
            
            return I_total

        # signed distance function
        def phi_n(q: jnp.ndarray, qb: jnp.ndarray, physics_params: dict) -> jnp.ndarray:
            """
            returns the signed distance function for point mass body with a box
            Args:
                `q`[jnp.ndarray]: pose of ee
                `qb`[jnp.ndarray]: pose of block
            """
            def softmax(a, b, eps=0.001):
                """
                softmax activation
                """
                return 0.5 * (a + b + jnp.sqrt((a-b)**2 + eps**2))
            def R(angle: float):
                """
                z-Rotation matrix
                """
                return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                                [jnp.sin(angle), jnp.cos(angle)]]).reshape((2,2))
            
            ## sdf1 (top)

            # split up the state vector of box
            qbx, qby, qbphi = jnp.split(qb, 3)
            # sdf vector
            phix1, phiy1 = jnp.split(
                jnp.abs(
                R(-qbphi)@(q - jnp.hstack([qbx, qby]))
                ) - jnp.hstack(
                    [
                        0.5*self.T_width + self.r_end_eff,
                        0.5*self.T_thickness + self.r_end_eff
                        # 0.5*physics_params['T_thickness'] + self.r_end_eff
                    ]
                ),
                2
            )
            sdf_top = (softmax(phix1,0.0)**2 + softmax(phiy1,0.0)**2)**0.5 \
                            -softmax(-softmax(phix1,phiy1),0.0)
            
            ## sdf2 (stem)

            dist_to_stem = 0.5*self.T_height

            # sdf vector
            phix2, phiy2 = jnp.split(
                jnp.abs(
                R(-qbphi)@(q - jnp.hstack([qbx + dist_to_stem*jnp.sin(qbphi), qby - dist_to_stem*jnp.cos(qbphi)]))
                ) - jnp.hstack(
                    [
                        0.5*self.T_thickness + self.r_end_eff,
                        # 0.5*physics_params['T_thickness'] + self.r_end_eff,
                        0.5*self.T_height + self.r_end_eff
                    ]
                ),
                2
            )
            sdf_stem = (softmax(phix2,0.0)**2 + softmax(phiy2,0.0)**2)**0.5 \
                            -softmax(-softmax(phix2,phiy2),0.0)
                
            return jnp.minimum(sdf_stem,sdf_top)

        # contact jacobian
        def J_end_eff_to_block(p_end_eff, p_block, physics_params: dict):
            return jnp.hstack(jacfwd(phi_n,argnums=0)(p_end_eff, p_block, physics_params)).reshape((1,2))
                    
        def J_block_to_end_eff(p_end_eff, p_block, physics_params: dict):
            return jnp.hstack(jacfwd(phi_n, argnums=1)(p_end_eff, p_block, physics_params)).reshape((1,3))

        def J_block_to_ground(v_block, physics_params):
            return (jnp.sign(v_block)*jnp.hstack([1.0,1.0,0.2116404183200049])).reshape((1,3))

        # contact model
        def get_contact_force(_phi_n, _contact_vel, beta = 3.0):
            impulse_normal = 1.0/beta * jnp.log(
                1.0 + jnp.exp(beta*(-self.contact_model_params['K']*_phi_n-self.contact_model_params['C']*_contact_vel))
            )
            return impulse_normal

        def get_ground_friction_force(weight_force, _contact_vel, physics_params, gamma = 0.1, k = 5.0): 
            return -physics_params['mu']*weight_force*jnp.tanh(k*_contact_vel)

        def get_forces(x: dict, physics_params: dict):
            # split up the state vectors
            p_ee1, v_ee1        = jnp.split(x['ee1'],   2)
            p_ee2, v_ee2        = jnp.split(x['ee2'],   2)
            p_block, v_block    = jnp.split(x['block'], 2)

            # get forces
            force_ee1 = get_contact_force(phi_n(p_ee1,p_block,physics_params), J_end_eff_to_block(p_ee1,p_block,physics_params)@v_ee1)
            force_ee2 = get_contact_force(phi_n(p_ee2,p_block,physics_params), J_end_eff_to_block(p_ee2,p_block,physics_params)@v_ee2)
            # force_gnd = get_ground_friction_force(physics_params['mass_block']*self.g, J_block_to_ground(v_block,physics_params)@v_block, physics_params)

            return {
                'ee1': force_ee1,
                'ee2': force_ee2
            }
        
        # dynamics model
        def f(x : dict, u : jnp.ndarray, physics_params : dict) -> dict:
            # split up the state vectors
            p_ee1, v_ee1        = jnp.split(x['ee1'],   2)
            p_ee2, v_ee2        = jnp.split(x['ee2'],   2)
            p_block, v_block    = jnp.split(x['block'], 2)

            # get contact force 
            force_ee1 = get_contact_force(phi_n(p_ee1,p_block,physics_params), J_end_eff_to_block(p_ee1,p_block,physics_params)@v_ee1)
            force_ee2 = get_contact_force(phi_n(p_ee2,p_block,physics_params), J_end_eff_to_block(p_ee2,p_block,physics_params)@v_ee2)
            force_gnd = get_ground_friction_force(physics_params['mass_block']*self.g, J_block_to_ground(v_block,physics_params)@v_block, physics_params)

            # define mass matrix of block and end_eff
            mass_end_eff_mat = jnp.eye(2)*self.mass_end_eff
            mass_block_mat = jnp.array([[physics_params['mass_block'],0.0,0.0],
                                        [0.0,physics_params['mass_block'],0.0],
                                        [0.0,0.0,physics_params['inertia']]])
                
            return {
                'ee1': jnp.hstack(
                    [
                        jnp.clip(p_ee1 + self.dt * v_ee1, self.pos_bounds[0], self.pos_bounds[1]).flatten(),
                        jnp.clip(v_ee1 + self.dt * jnp.linalg.inv(mass_end_eff_mat) @ 
                         (J_end_eff_to_block(p_ee1,p_block,physics_params).T@force_ee1 + u['ee1']), self.vel_bounds[0], self.vel_bounds[1]).flatten()
                        # jnp.clip(v_ee1 + self.dt * jnp.linalg.inv(mass_end_eff_mat) @ u['ee1'], self.vel_bounds[0], self.vel_bounds[1]).flatten()
                    ]
                ),
                'ee2': jnp.hstack(
                    [
                        jnp.clip(p_ee2 + self.dt * v_ee2, self.pos_bounds[0], self.pos_bounds[1]).flatten(),
                        jnp.clip(v_ee2 + self.dt * jnp.linalg.inv(mass_end_eff_mat) @ 
                         (J_end_eff_to_block(p_ee2,p_block,physics_params).T@force_ee2 + u['ee2']), self.vel_bounds[0], self.vel_bounds[1]).flatten()
                        # jnp.clip(v_ee2 + self.dt * jnp.linalg.inv(mass_end_eff_mat) @ u['ee2'], self.vel_bounds[0], self.vel_bounds[1]).flatten()
                    ]
                ),
                'block': jnp.hstack(
                    [
                        (p_block + self.dt * v_block).flatten(),
                        (v_block + self.dt * jnp.linalg.inv(mass_block_mat) @ 
                         (J_block_to_end_eff(p_ee1,p_block,physics_params).T@force_ee1
                          + J_block_to_end_eff(p_ee2,p_block,physics_params).T@force_ee2 
                          + J_block_to_ground(v_block,physics_params).T@force_gnd)).flatten()
                    ]
                )
            }
        
        def measure(key, state: dict):
            _keys = jax.random.split(key, 3)
            return {'ee1': state['ee1'] + self.robot_sensor_variance**2.0 * jax.random.normal(_keys[0],shape=state['ee1'].shape),
                    'ee2': state['ee2'] + self.robot_sensor_variance**2.0 * jax.random.normal(_keys[1],shape=state['ee2'].shape),
                    'block': state['block'] + self.block_tag_sensor_variance**2.0 * jax.random.normal(_keys[2],shape=state['block'].shape)}

        @jax.jit
        def step(key: jnp.ndarray, x_true : dict, u : jnp.ndarray) -> dict:

            # TRUE PHYSICS PARAMETERS 
            true_physics_params = {
                                'inertia': self.get_inertia_T(),
                                'mass_block': self.mass_block,
                                'mu': self.ground_contact_model_params['mu'],
                                # 'T_thickness': self.T_thickness
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
        def rollout(x_init : dict, us : jnp.ndarray, physics_params : dict):
            return jax.lax.scan(lambda x,u : (f(x,u,physics_params), x), init=x_init, xs=us, length=us['ee1'].shape[0])[-1]

        def check_goal(goal, state) -> bool:
            distance = jnp.abs(state['block'] - goal)
            if jnp.sum(distance) <= 0.1:
                return True, distance[:3].flatten()
            return False, distance[:3].flatten()

        def animate_simulation(states_stack, trajsplines_stack, num_param_samples, experiment_string, idx):
            """Animate a completed experiment (requires the optional `animate` module)."""
            try:
                from .animate import animate
            except ImportError as e:
                raise ImportError(
                    'Animation requires the optional `animate` module in '
                    'config/config_bimanual/. Install or add it to enable visualization.'
                ) from e
            return animate(states_stack, trajsplines_stack, num_param_samples, experiment_string, idx)
        
        self.get_inertia_T =                get_inertia_T
        self.phi_n =                        phi_n
        self.J_block_to_end_eff =           J_block_to_end_eff
        self.J_block_to_ground =            J_block_to_ground
        self.J_end_eff_to_block =           J_end_eff_to_block
        self.get_contact_force =            get_contact_force
        self.get_ground_friction_force =    get_ground_friction_force
        self.f =                            f
        self.step =                         step
        self.sim_fwd =                      sim_fwd
        self.rollout =                      rollout
        self.check_goal =                   check_goal
        self.animate_simulation =           animate_simulation
        