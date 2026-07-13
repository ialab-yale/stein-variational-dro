"""
Franka Panda + rigidly mounted waiter tray + sliding block visualization.
"""

import os
import pickle
import time
import warnings
import dill as pkl

warnings.filterwarnings("ignore", message=r"os\.fork\(\) was called", category=RuntimeWarning)

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import imageio


def render_waiter(idx: int, method: str, verbose: bool = True) -> str:
    """Render one experiment trial to MP4.

    Args:
        idx: trial index of the saved trajectory.
        method: planning method name, e.g. 'SVDRO', 'EMPPI', 'MPC', 'DRO'.
        verbose: if False, suppress all progress printing.

    Returns:
        Path of the written MP4 file.
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    log(
        "============================================================================="
        f"  {method}  idx={idx}  "
        "============================================================================="
    )
    # =============================================================================
    # Paths
    # =============================================================================

    PANDA_DIR = "../mujoco_menagerie/franka_emika_panda"
    PANDA_XML = os.path.join(PANDA_DIR, "panda_nohand.xml")

    DATA_PATH = f"data/dynamic_waiter/pkl/traj/{method}_states-idx{idx}.pkl"
    SCENE_PARAMS_PATH = "data/dynamic_waiter/pkl/scene/scene_data.pkl"
    OUT_PATH = f"videos/dynamic_waiter/mp4/{method}_franka_vis_idx{idx}.mp4"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # =============================================================================
    # Load scene params
    # =============================================================================

    with open(SCENE_PARAMS_PATH, 'rb') as pickle_file:
        scene_params = pkl.load(pickle_file)

    # =============================================================================
    # Trajectory loading
    # =============================================================================

    with open(DATA_PATH, "rb") as f:
        raw = pickle.load(f)

    TO_TIME = 750*10
    t_data = np.asarray(raw["t"], dtype=float)[:TO_TIME]
    b_data = np.asarray(raw["b"], dtype=float)[:TO_TIME]

    STRIDE = 10
    t_data = t_data[::STRIDE]
    b_data = b_data[::STRIDE]
    N = len(t_data)

    log(f"Trajectory: {N} frames @ stride {STRIDE}")
    log(f"t_data shape: {t_data.shape}")
    log(f"b_data shape: {b_data.shape}")


    # =============================================================================
    # Geometry parameters
    # =============================================================================
    TRAY_R = scene_params['table_width'] / 2.0 
    TRAY_T = scene_params['table_height'] * 0.05

    BLOCK_LW = scene_params['block_width'] / 2.0
    BLOCK_H = scene_params['block_height'] / 2.0

    # z-offset of the block center above the tray center
    BZ = TRAY_T / 2.0 + BLOCK_H / 2.0

    # Shift the whole tray-block system in world coordinates.
    SYSTEM_SHIFT = np.array([0.0, 0.5, -0.15], dtype=float)


    # =============================================================================
    # Initial arm configuration
    # =============================================================================

    Q0 = np.array(
        [1.92903, -0.92005, -0.44644, -1.32956, 2.0126, 2.62838, 0.20088],
        dtype=float,
    )


    # =============================================================================
    # Helpers
    # =============================================================================

    def normalize(v, eps=1e-12):
        n = np.linalg.norm(v)
        if n < eps:
            return v.copy()
        return v / n


    def so3_error(R_des, R_cur):
        """
        Orientation error that rotates R_cur toward R_des.

        Both R_des and R_cur are 3x3 rotation matrices expressed in world frame.

        Returns:
            3-vector angular velocity-like error.
        """
        return 0.5 * (
            np.cross(R_cur[:, 0], R_des[:, 0])
            + np.cross(R_cur[:, 1], R_des[:, 1])
            + np.cross(R_cur[:, 2], R_des[:, 2])
        )


    def make_rotation_with_world_z_from_initial(R0):
        """
        Build a desired orientation whose local z-axis is exactly world +z, while
        preserving the initial yaw direction as much as possible.

        This is useful if the initial end-effector is visually correct but you want
        to guarantee the tray normal is exactly global +z.
        """
        z_des = np.array([0.0, 0.0, 1.0])

        # Project the initial local x-axis into the horizontal plane.
        x0 = R0[:, 0].copy()
        x_proj = x0 - np.dot(x0, z_des) * z_des

        if np.linalg.norm(x_proj) < 1e-8:
            # Fallback if the initial x-axis is almost vertical.
            y0 = R0[:, 1].copy()
            x_proj = y0 - np.dot(y0, z_des) * z_des

        x_des = normalize(x_proj)
        y_des = normalize(np.cross(z_des, x_des))
        x_des = normalize(np.cross(y_des, z_des))

        R_des = np.column_stack([x_des, y_des, z_des])
        return R_des


    def find_body(spec, name):
        for body in spec.bodies:
            if body.name == name:
                return body
        raise RuntimeError(f"Could not find body '{name}' in MjSpec.")


    def find_site_in_body(body, name):
        for site in body.sites:
            if site.name == name:
                return site
        raise RuntimeError(f"Could not find site '{name}' in body '{body.name}'.")


    # =============================================================================
    # Compute initial end-effector pose from plain Panda model
    # =============================================================================

    _m0 = mujoco.MjModel.from_xml_path(PANDA_XML)
    _d0 = mujoco.MjData(_m0)

    _d0.qpos[:7] = Q0
    mujoco.mj_forward(_m0, _d0)

    _sid0 = mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if _sid0 < 0:
        raise RuntimeError("Could not find site 'attachment_site' in panda_nohand.xml.")

    EE_BASE = _d0.site_xpos[_sid0].copy()
    EE_ROT0 = _d0.site_xmat[_sid0].reshape(3, 3).copy()

    del _m0, _d0

    log(f"Initial attachment_site position EE_BASE: {EE_BASE.round(5)}")
    log(f"Initial attachment_site z-axis: {EE_ROT0[:, 2].round(5)}")

    # Desired orientation:
    #
    # TRUE:
    #   Use the exact initial site orientation.
    #   This is usually the safest choice if the initial visualization starts upright.
    #
    # FALSE:
    #   Force the site local z-axis to be exactly world +z, while preserving yaw
    #   from the initial orientation.
    USE_EXACT_INITIAL_ORIENTATION = False

    if USE_EXACT_INITIAL_ORIENTATION:
        R_DES = EE_ROT0.copy()
    else:
        R_DES = make_rotation_with_world_z_from_initial(EE_ROT0)

    log(f"Desired site z-axis: {R_DES[:, 2].round(5)}")


    # =============================================================================
    # Initial block pose
    # =============================================================================

    # The tray center at frame 0 is EE_BASE + [t0_x, t0_y, 0].
    # Since target below is also EE_BASE + [t_x, t_y, 0], the block should be placed
    # as tray_center + [b_x - t_x, b_y - t_y, BZ].
    brel0 = np.array(
        [
            b_data[0, 1] - t_data[0, 1],
            b_data[0, 0] - t_data[0, 0],
            BZ,
        ],
        dtype=float,
    )

    tray0 = EE_BASE + SYSTEM_SHIFT + np.array([t_data[0, 1], t_data[0, 0], 0.0])
    block_w0 = tray0 + brel0


    # =============================================================================
    # Build Panda spec and rigidly attach tray
    # =============================================================================

    panda_spec = mujoco.MjSpec.from_file(PANDA_XML)

    attach_body = find_body(panda_spec, "attachment")

    # The tray should be mounted at the attachment site if possible.
    # If the site exists in this body, use its local pose. This avoids accidentally
    # mounting the tray at the attachment body origin instead of the actual EE site.
    try:
        attach_site = find_site_in_body(attach_body, "attachment_site")
        tray_local_pos = np.array(attach_site.pos, dtype=float).copy()
        tray_local_quat = np.array(attach_site.quat, dtype=float).copy()
        log("Mounting tray at local pose of attachment_site.")
    except Exception as e:
        log(f"Warning: {e}")
        log("Falling back to mounting tray at attachment body origin.")
        tray_local_pos = np.zeros(3)
        tray_local_quat = np.array([1.0, 0.0, 0.0, 0.0])

    tray = attach_body.add_body()
    tray.name = "tray"
    tray.pos[:] = tray_local_pos
    tray.quat[:] = tray_local_quat

    # Tray disk.
    gd = tray.add_geom()
    gd.name = "tray_disk"
    gd.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    gd.size[0] = TRAY_R
    gd.size[1] = TRAY_T / 2.0
    gd.rgba[:] = [0.62, 0.63, 0.65, 1.0]
    gd.contype = 0
    gd.conaffinity = 0

    # Tray rim.
    gr = tray.add_geom()
    gr.name = "tray_rim"
    gr.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    gr.size[0] = TRAY_R + 0.005
    gr.size[1] = TRAY_T / 2.0 + 0.003
    gr.rgba[:] = [0.52, 0.53, 0.55, 1.0]
    gr.contype = 0
    gr.conaffinity = 0

    # Tray frame
    tray_frame = tray.add_body()
    tray_frame.name = "tray_frame"
    tray_frame.pos[:] = [0.0, 0.0, BLOCK_H/2.0]

    # Rotate tray about local z-axis to match block axes orientation
    yaw_offset = np.deg2rad(-180-10)   
    tray_frame.quat[:] = [
        np.cos(yaw_offset / 2.0),
        0.0,
        0.0,
        np.sin(yaw_offset / 2.0),
    ]

    gx = tray_frame.add_geom()
    gx.name = "tray_x_axis"
    gx.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    gx.fromto[:] = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    gx.size[0] = 0.01
    gx.rgba[:] = [1.0, 0.647, 0.0, 1.0]
    gx.contype = 0
    gx.conaffinity = 0

    gy = tray_frame.add_geom()
    gy.name = "tray_y_axis"
    gy.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    gy.fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    gy.size[0] = 0.01
    gy.rgba[:] = [1.0, 0.647, 0.0, 1.0]
    gy.contype = 0
    gy.conaffinity = 0

    gz = tray_frame.add_geom()
    gz.name = "tray_z_axis"
    gz.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    gz.fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    gz.size[0] = 0.01
    gz.rgba[:] = [1.0, 0.647, 0.0, 1.0]
    gz.contype = 0

    # =============================================================================
    # Progressive block trajectory markers
    # =============================================================================

    TRAIL_EVERY = 5          # place one marker every 5 rendered frames
    TRAIL_RADIUS = 0.01
    MAX_TRAIL_MARKERS = int(np.ceil(N / TRAIL_EVERY))

    traj_marker_xml = ""
    for k in range(MAX_TRAIL_MARKERS):
        traj_marker_xml += f"""
        <body name="block_traj_marker_{k}" mocap="true" pos="0 0 -10">
        <geom name="block_traj_marker_geom_{k}"
                type="sphere"
                size="{TRAIL_RADIUS}"
                material="traj_marker"
                contype="0"
                conaffinity="0"/>
        </body>
        """

    # =============================================================================
    # Outer scene with block mocap body
    # =============================================================================

    scene_xml = f"""
    <mujoco model="franka_tray_block">
    <compiler angle="radian" autolimits="true"/>
    <option timestep="0.002"/>

    <visual>
        <quality shadowsize="4096" offsamples="8"/>
        <global offwidth="1280" offheight="720"/>
    </visual>

    <asset>
        <texture name="chk" type="2d" builtin="checker"
                rgb1="0.20 0.20 0.22"
                rgb2="0.28 0.28 0.30"
                width="512"
                height="512"/>
        <material name="floor_mat"
                texture="chk"
                texrepeat="6 6"
                reflectance="0.04"
                specular="0.05"/>
        <material name="blk_mat"
                rgba="0.09 0.47 0.55 0.3"
                specular="0.7"
                shininess="0.6"/>
        <material name="frame_x_mat" rgba="1 0 0 1"/>
        <material name="frame_y_mat" rgba="1 0 0 1"/>
        <material name="frame_z_mat" rgba="1 0 0 1"/>
        <material name="traj_marker" rgba="0.9 0.9 0.9 0.25"/>
    </asset>

    <worldbody>
        <light name="key"
            pos="1.2 1.8 3.5"
            dir="-0.3 -0.5 -1"
            diffuse="0.88 0.84 0.78"
            specular="0.35 0.35 0.30"
            castshadow="false"/>
        <light name="fill"
            pos="-2.0 -1.2 3.0"
            dir="0.4 0.3 -1"
            diffuse="0.30 0.32 0.38"
            castshadow="false"/>
        <light name="rim"
            pos="0 -2.5 1.8"
            dir="0 0.6 -0.4"
            diffuse="0.18 0.18 0.22"
            castshadow="false"/>

        <geom name="floor"
            type="plane"
            pos="0 0 0"
            size="3 3 0.1"
            material="floor_mat"
            conaffinity="0"
            contype="0"/>

        {traj_marker_xml}

        <body name="block_mocap"
            mocap="true"
            pos="{block_w0[0]:.6f} {block_w0[1]:.6f} {block_w0[2]:.6f}">
        <geom name="block_geom"
                type="box"
                size="{BLOCK_LW / 2.0} {BLOCK_LW / 2.0} {BLOCK_H / 2.0}"
                material="blk_mat"
                conaffinity="0"
                contype="0"/>
        <body name="block_frame" pos="0 0 0">
            <geom name="block_x_axis"
                type="capsule"
                fromto="0 0 0  0.1 0 0"
                size="0.01"
                material="frame_x_mat"
                contype="0"
                conaffinity="0"/>

            <geom name="block_y_axis"
                type="capsule"
                fromto="0 0 0  0 0.1 0"
                size="0.01"
                material="frame_y_mat"
                contype="0"
                conaffinity="0"/>

            <geom name="block_z_axis"
                type="capsule"
                fromto="0 0 0  0 0 0.1"
                size="0.01"
                material="frame_z_mat"
                contype="0"
                conaffinity="0"/>
        </body>
        </body>
    </worldbody>
    </mujoco>
    """

    scene_spec = mujoco.MjSpec.from_string(scene_xml)

    frame = scene_spec.worldbody.add_frame()
    scene_spec.attach(panda_spec, prefix="panda/", suffix="", frame=frame)

    model = scene_spec.compile()
    data = mujoco.MjData(model)

    log(f"Model: nq={model.nq}, nv={model.nv}, nbody={model.nbody}, nmocap={model.nmocap}")


    # =============================================================================
    # IDs
    # =============================================================================

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "panda/attachment_site",
    )
    if site_id < 0:
        raise RuntimeError("Could not find site 'panda/attachment_site' after attaching Panda.")

    blk_bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "block_mocap",
    )
    if blk_bid < 0:
        raise RuntimeError("Could not find body 'block_mocap'.")

    blk_mid = model.body_mocapid[blk_bid]
    if blk_mid < 0:
        raise RuntimeError("'block_mocap' is not registered as a mocap body.")

    traj_marker_mids = []
    for k in range(MAX_TRAIL_MARKERS):
        bid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"block_traj_marker_{k}",
        )
        if bid < 0:
            raise RuntimeError(f"Could not find block_traj_marker_{k}.")
        traj_marker_mids.append(model.body_mocapid[bid])

    jnt_ids = []
    for i in range(7):
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"panda/joint{i + 1}",
        )
        if jid < 0:
            raise RuntimeError(f"Could not find joint 'panda/joint{i + 1}'.")
        jnt_ids.append(jid)

    jnt_addr = [model.jnt_qposadr[j] for j in jnt_ids]
    dof_ids = np.array([model.jnt_dofadr[j] for j in jnt_ids], dtype=int)

    Q_LO = np.array([model.jnt_range[j, 0] for j in jnt_ids], dtype=float)
    Q_HI = np.array([model.jnt_range[j, 1] for j in jnt_ids], dtype=float)
    Q_MID = 0.5 * (Q_LO + Q_HI)


    # =============================================================================
    # Initialize arm
    # =============================================================================

    for k, a in enumerate(jnt_addr):
        data.qpos[a] = Q0[k]

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    log("Initial compiled site position:", data.site_xpos[site_id].round(5))
    log("Initial compiled site z-axis:", data.site_xmat[site_id].reshape(3, 3)[:, 2].round(5))


    # =============================================================================
    # IK constants
    # =============================================================================

    DAMP = 0.006          # lower damping = more exact pose tracking
    ALPHA = 0.10          # smaller step = less overshoot / less residual tilt
    NULL_GAIN = 0.0       # turn this off; it fights exact orientation tracking

    MAX_ITER = 100        # give offline IK enough iterations

    POS_TOL = 2e-4
    ROT_TOL = 1e-6        # much stricter orientation tolerance

    POS_GAIN = 1.0
    ROT_GAIN = 25.0       # make orientation dominate visually

    MAX_DQ = 0.025        # smaller updates give cleaner final orientation


    # =============================================================================
    # Precompute 6D pose IK trajectory
    # =============================================================================

    log("Pre-computing 6D pose IK solutions...")

    q_traj = np.zeros((N, 7), dtype=float)
    ee_pos_errs_mm = []
    ee_rot_errs = []
    tray_normal_errs_deg = []

    t_start = time.time()

    # Start from Q0.
    q_prev = Q0.copy()

    # initial tray state definition
    t0_vis = np.array([t_data[0, 1], t_data[0, 0]], dtype=float)

    for i in range(N):
        tx = float(t_data[i, 1])
        ty = float(t_data[i, 0])

        # shifted pose
        t_vis = np.array([tx, ty], dtype=float)
        dt_vis = t_vis - t0_vis

        # define the target pose
        target_pos = EE_BASE + SYSTEM_SHIFT + np.array([dt_vis[0], dt_vis[1], 0.0], dtype=float)

        # Warm start from previous frame.
        q = q_prev.copy()

        for it in range(MAX_ITER):
            for k, a in enumerate(jnt_addr):
                data.qpos[a] = q[k]

            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            cur_pos = data.site_xpos[site_id].copy()
            R_cur = data.site_xmat[site_id].reshape(3, 3).copy()

            ep = target_pos - cur_pos
            er = so3_error(R_DES, R_cur)

            pos_err = np.linalg.norm(ep)
            rot_err = np.linalg.norm(er)

            if pos_err < POS_TOL and rot_err < ROT_TOL:
                break

            jacp = np.zeros((3, model.nv), dtype=float)
            jacr = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

            Jp = jacp[:, dof_ids]
            Jr = jacr[:, dof_ids]

            J = np.vstack(
                [
                    POS_GAIN * Jp,
                    ROT_GAIN * Jr,
                ]
            )

            e = np.concatenate(
                [
                    POS_GAIN * ep,
                    ROT_GAIN * er,
                ]
            )

            # Damped least-squares pseudoinverse:
            # dq = J^T (J J^T + lambda^2 I)^-1 e
            JJt = J @ J.T + (DAMP**2) * np.eye(6)
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))

            dq_task = Jpinv @ e

            # Nullspace posture regularization.
            Nproj = np.eye(7) - Jpinv @ J
            dq_null = Nproj @ (NULL_GAIN * (Q_MID - q))

            dq = ALPHA * (dq_task + dq_null)
            dq = np.clip(dq, -MAX_DQ, MAX_DQ)

            q = np.clip(q + dq, Q_LO, Q_HI)

        # Store final solution.
        for k, a in enumerate(jnt_addr):
            data.qpos[a] = q[k]

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        q_traj[i] = q
        q_prev = q.copy()

        final_pos = data.site_xpos[site_id].copy()
        R_final = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err_mm = np.linalg.norm(final_pos - target_pos) * 1000.0
        rot_err = np.linalg.norm(so3_error(R_DES, R_final))

        tray_z = R_final[:, 2]
        vertical_cos = np.clip(np.dot(normalize(tray_z), np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        normal_err_deg = np.degrees(np.arccos(vertical_cos))

        ee_pos_errs_mm.append(pos_err_mm)
        ee_rot_errs.append(rot_err)
        tray_normal_errs_deg.append(normal_err_deg)

        if i % 100 == 0 or i == N - 1:
            log(
                f"  IK [{i:4d}/{N}] "
                f"pos_err={pos_err_mm:8.3f} mm | "
                f"rot_err={rot_err:8.5f} | "
                f"tray_z={tray_z.round(4)} | "
                f"normal_err={normal_err_deg:7.3f} deg | "
                f"iters={it + 1:3d} | "
                f"elapsed={time.time() - t_start:.1f}s"
            )

    log("IK done.")
    log(f"  Mean position error: {np.mean(ee_pos_errs_mm):.3f} mm")
    log(f"  Max  position error: {np.max(ee_pos_errs_mm):.3f} mm")
    log(f"  Mean rotation error: {np.mean(ee_rot_errs):.6f}")
    log(f"  Max  rotation error: {np.max(ee_rot_errs):.6f}")
    log(f"  Mean tray normal error from world +z: {np.mean(tray_normal_errs_deg):.4f} deg")
    log(f"  Max  tray normal error from world +z: {np.max(tray_normal_errs_deg):.4f} deg")


    # =============================================================================
    # Camera
    # =============================================================================

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cam.lookat[:] = [
        EE_BASE[1] + SYSTEM_SHIFT[1] + float(np.median(t_data[:, 1])),
        EE_BASE[0] + SYSTEM_SHIFT[0],
        EE_BASE[2] + SYSTEM_SHIFT[2] + float(np.median(t_data[:, 1])),
    ]

    cam.distance = 1.8
    cam.azimuth = 0.0       # side profile
    cam.elevation = 0.0     # horizontal side profile


    # =============================================================================
    # Rendering
    # =============================================================================

    W, H = 1280, 720
    FPS = 30

    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()

    frames = []

    log("Rendering frames...")
    t_render = time.time()

    # redefine tray state
    t0_vis = np.array([t_data[0, 1], t_data[0, 0]], dtype=float)

    for i in range(N):
        tx = float(t_data[i, 1])
        ty = float(t_data[i, 0])

        # Apply precomputed joint configuration.
        for k, a in enumerate(jnt_addr):
            data.qpos[a] = q_traj[i, k]

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Current tray center.
        tray_pos = data.site_xpos[site_id].copy()
        R_site = data.site_xmat[site_id].reshape(3, 3).copy()

        # Block placement.
        bx = float(b_data[i, 1])
        by = float(b_data[i, 0])
        btheta = float(b_data[i, 2])

        block_rel = np.array(
            [
                bx - tx - 0.1,
                by - ty,
                BZ,
            ],
            dtype=float,
        )

        data.mocap_pos[blk_mid] = tray_pos + block_rel

        # Progressive transparent red trail.
        if i % TRAIL_EVERY == 0:
            marker_idx = i // TRAIL_EVERY
            if marker_idx < MAX_TRAIL_MARKERS:
                data.mocap_pos[traj_marker_mids[marker_idx]] = data.mocap_pos[blk_mid].copy()
                data.mocap_quat[traj_marker_mids[marker_idx]] = np.array([1.0, 0.0, 0.0, 0.0])

        half = 0.5 * btheta
        data.mocap_quat[blk_mid] = np.array(
            [
                np.cos(half),
                0.0,
                np.sin(half),
                0.0,
            ],
            dtype=float,
        )

        mujoco.mj_forward(model, data)

        if i % 100 == 0 or i == N - 1:
            tray_z = R_site[:, 2]
            log(
                f"  Render [{i:4d}/{N}] "
                f"tray_pos={tray_pos.round(4)} | "
                f"tray_z={tray_z.round(4)} | "
                f"block_pos={data.mocap_pos[blk_mid].round(4)} | "
                f"elapsed={time.time() - t_render:.1f}s"
            )

        renderer.update_scene(data, camera=cam, scene_option=opt)
        frame = renderer.render().copy()
        frames.append(frame)

    renderer.close()

    log(f"Render done in {time.time() - t_render:.1f}s")


    # =============================================================================
    # Write MP4
    # =============================================================================

    log(f"Writing video to {OUT_PATH}...")

    with imageio.get_writer(
        OUT_PATH,
        fps=FPS,
        quality=9,
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    log(
        f"\nVideo -> {OUT_PATH} "
        f"({len(frames) / FPS:.1f}s | {len(frames)} frames)"
    )

    return OUT_PATH


if __name__ == "__main__":
    for idx in range(32):
        for method in ['SVDRO', 'EMPPI', 'DuSTMPC', 'MPC', 'DRO']:
            render_waiter(idx, method)