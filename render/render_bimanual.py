"""Bimanual T-block visualization — two sphere end-effectors + T-shaped block."""

import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore", message=r"os\.fork\(\) was called", category=RuntimeWarning)

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import imageio
import dill as pkl


# =============================================================================
# Outer loop – methods × trajectory indices
# =============================================================================


def render_bimanual(idx: int, method: str, verbose: bool = True) -> str:
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

    DATA_PATH         = f"data/bimanual/pkl/traj/{method}_states-idx{idx}.pkl"
    SCENE_PARAMS_PATH = "data/bimanual/pkl/scene/scene_data.pkl"
    OUT_PATH          = f"videos/bimanual/mp4/{method}_vis_idx{idx}.mp4"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # =============================================================================
    # Load scene params
    # =============================================================================

    with open(SCENE_PARAMS_PATH, 'rb') as f:
        scene_params = pkl.load(f)

    # =============================================================================
    # Trajectory loading
    # =============================================================================

    with open(DATA_PATH, 'rb') as f:
        raw = pkl.load(f)

    TO_TIME = 750 * 10
    ee1_data = np.asarray(raw['ee1'],   dtype=float)[:TO_TIME, :2]   # (T, 2)  x, y
    ee2_data = np.asarray(raw['ee2'],   dtype=float)[:TO_TIME, :2]
    blk_data = np.asarray(raw['block'], dtype=float)[:TO_TIME, :3]   # (T, 3)  x, y, yaw

    STRIDE   = 10
    ee1_data = ee1_data[::STRIDE]
    ee2_data = ee2_data[::STRIDE]
    blk_data = blk_data[::STRIDE]
    N = len(blk_data)

    log(f"Trajectory: {N} frames @ stride {STRIDE}")

    # =============================================================================
    # Geometry parameters
    # =============================================================================

    T_w = scene_params['T_width']
    T_h = scene_params['T_height']
    T_t = scene_params['T_thickness']
    R_EE = scene_params['r_end_eff']

    # MuJoCo box half-extents for the T-block
    BAR_HX  = T_w / 2.0
    BAR_HY  = T_t / 2.0
    BAR_HZ  = T_t * 0.75 / 2.0

    STEM_HX = T_t / 2.0
    STEM_HY = T_h / 2.0
    STEM_HZ = T_t * 0.75 / 2.0

    # Stem Y offset: top face of stem flush with bottom face of bar
    STEM_OY = -(T_t / 2.0 + T_h / 2.0)

    # Fixed world Z for all planar objects (matches meshcat: T_thickness * 0.375)
    BLOCK_Z = T_t * 0.375
    EE_Z    = T_t * 0.375

    # Goal pose (static, semi-transparent T)
    GOAL_X, GOAL_Y, GOAL_YAW = 0.2, 0.2, np.pi / 4.0
    GOAL_QW = np.cos(GOAL_YAW / 2.0)
    GOAL_QZ = np.sin(GOAL_YAW / 2.0)

    # =============================================================================
    # Helpers
    # =============================================================================

    def yaw_to_quat(yaw):
        """Planar yaw about world Z → quaternion [w, x, y, z]."""
        return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

    def get_mocap_id(model, body_name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"Body '{body_name}' not found.")
        mid = model.body_mocapid[bid]
        if mid < 0:
            raise RuntimeError(f"Body '{body_name}' is not a mocap body.")
        return mid

    # =============================================================================
    # Progressive end-effector trajectory markers
    # =============================================================================

    TRAIL_EVERY       = 10
    TRAIL_RADIUS      = scene_params['r_end_eff'] / 2.0
    MAX_TRAIL_MARKERS = int(np.ceil(N / TRAIL_EVERY))

    traj_marker_xml = ""
    for k in range(MAX_TRAIL_MARKERS):
        traj_marker_xml += f"""
        <body name="ee1_traj_marker_{k}" mocap="true" pos="0 0 -10">
        <geom name="ee1_traj_marker_geom_{k}"
                type="sphere"
                size="{TRAIL_RADIUS}"
                material="traj_marker"
                contype="0"
                conaffinity="0"/>
        </body>

        <body name="ee2_traj_marker_{k}" mocap="true" pos="0 0 -10">
        <geom name="ee2_traj_marker_geom_{k}"
                type="sphere"
                size="{TRAIL_RADIUS}"
                material="traj_marker"
                contype="0"
                conaffinity="0"/>
        </body>
        """

    # =============================================================================
    # Scene XML
    # =============================================================================

    scene_xml = f"""
    <mujoco model="bimanual_T_block">
      <compiler angle="radian" autolimits="true"/>
      <option gravity="0 0 0" timestep="0.002"/>

      <visual>
        <quality shadowsize="4096" offsamples="8"/>
        <global offwidth="1280" offheight="720"/>
      </visual>

      <asset>
        <texture name="chk" type="2d" builtin="checker"
                 rgb1="0.20 0.20 0.22" rgb2="0.28 0.28 0.30"
                 width="512" height="512"/>
        <material name="floor_mat"
                  texture="chk" texrepeat="6 6"
                  reflectance="0.04" specular="0.05"/>
        <material name="block_mat"
                  rgba="1.0 0.498 0.055 1.0"
                  specular="0.4" shininess="0.3"/>
        <material name="goal_mat"
                  rgba="1.0 0.498 0.055 0.2"/>
        <material name="ee_mat"
                  rgba="0.43 0.43 0.43 1.0"/>
        <material name="traj_marker"
                  rgba="0.9 0.9 0.9 0.1"/>
      </asset>

      <worldbody>
        <light name="key"  pos="1.2 1.8 3.5"  dir="-0.3 -0.5 -1"
               diffuse="0.88 0.84 0.78" specular="0.35 0.35 0.30" castshadow="false"/>
        <light name="fill" pos="-2.0 -1.2 3.0" dir="0.4 0.3 -1"
               diffuse="0.30 0.32 0.38" castshadow="false"/>
        <light name="rim"  pos="0 -2.5 1.8"   dir="0 0.6 -0.4"
               diffuse="0.18 0.18 0.22" castshadow="false"/>

        <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1"
              material="floor_mat" contype="0" conaffinity="0"/>

        <!-- Block trajectory trail markers -->
        {traj_marker_xml}
        
        <!-- Goal T: static, semi-transparent -->
        <body name="goal_T"
              pos="{GOAL_X} {GOAL_Y} {BLOCK_Z}"
              quat="{GOAL_QW:.6f} 0 0 {GOAL_QZ:.6f}">
          <geom name="goal_bar"  type="box"
                size="{BAR_HX}  {BAR_HY}  {BAR_HZ}"
                material="goal_mat" contype="0" conaffinity="0"/>
          <geom name="goal_stem" type="box"
                size="{STEM_HX} {STEM_HY} {STEM_HZ}"
                pos="0 {STEM_OY:.6f} 0"
                material="goal_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- T-block: two-box mocap body -->
        <body name="block_mocap" mocap="true"
              pos="{blk_data[0,0]:.6f} {blk_data[0,1]:.6f} {BLOCK_Z:.6f}">
          <geom name="block_bar"  type="box"
                size="{BAR_HX}  {BAR_HY}  {BAR_HZ}"
                material="block_mat" contype="0" conaffinity="0"/>
          <geom name="block_stem" type="box"
                size="{STEM_HX} {STEM_HY} {STEM_HZ}"
                pos="0 {STEM_OY:.6f} 0"
                material="block_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- End-effector spheres: mocap bodies -->
        <body name="ee1_mocap" mocap="true"
              pos="{ee1_data[0,0]:.6f} {ee1_data[0,1]:.6f} {EE_Z:.6f}">
          <geom name="ee1_geom" type="sphere" size="{R_EE}"
                material="ee_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="ee2_mocap" mocap="true"
              pos="{ee2_data[0,0]:.6f} {ee2_data[0,1]:.6f} {EE_Z:.6f}">
          <geom name="ee2_geom" type="sphere" size="{R_EE}"
                material="ee_mat" contype="0" conaffinity="0"/>
        </body>

      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(scene_xml)
    data  = mujoco.MjData(model)

    log(f"Model: nbody={model.nbody}, nmocap={model.nmocap}")

    # =============================================================================
    # IDs
    # =============================================================================

    blk_mid  = get_mocap_id(model, "block_mocap")
    ee1_mid  = get_mocap_id(model, "ee1_mocap")
    ee2_mid  = get_mocap_id(model, "ee2_mocap")

    ee1_traj_marker_mids = [
        get_mocap_id(model, f"ee1_traj_marker_{k}")
        for k in range(MAX_TRAIL_MARKERS)
    ]

    ee2_traj_marker_mids = [
        get_mocap_id(model, f"ee2_traj_marker_{k}")
        for k in range(MAX_TRAIL_MARKERS)
    ]

    # =============================================================================
    # Camera
    # =============================================================================

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cam.lookat[:] = [
        float(np.median(blk_data[:, 0])),
        float(np.median(blk_data[:, 1])),
        BLOCK_Z,
    ]
    cam.distance  = 2.35
    cam.azimuth   = -45.0
    cam.elevation = -35.264

    # =============================================================================
    # Rendering
    # =============================================================================

    W, H = 1280, 720
    FPS  = 30

    renderer = mujoco.Renderer(model, height=H, width=W)
    opt      = mujoco.MjvOption()
    frames   = []

    log("Rendering frames...")
    t_render = time.time()

    for i in range(N):

        # T-block pose
        data.mocap_pos[blk_mid]  = [blk_data[i, 0], blk_data[i, 1], BLOCK_Z]
        data.mocap_quat[blk_mid] = yaw_to_quat(blk_data[i, 2])

        # End-effector positions (spheres, no rotation)
        data.mocap_pos[ee1_mid]  = [ee1_data[i, 0], ee1_data[i, 1], EE_Z]
        data.mocap_quat[ee1_mid] = [1.0, 0.0, 0.0, 0.0]
        data.mocap_pos[ee2_mid]  = [ee2_data[i, 0], ee2_data[i, 1], EE_Z]
        data.mocap_quat[ee2_mid] = [1.0, 0.0, 0.0, 0.0]

        # Progressive end-effector trails
        if i % TRAIL_EVERY == 0:
            marker_idx = i // TRAIL_EVERY
            if marker_idx < MAX_TRAIL_MARKERS:
                data.mocap_pos[ee1_traj_marker_mids[marker_idx]] = data.mocap_pos[ee1_mid].copy()
                data.mocap_quat[ee1_traj_marker_mids[marker_idx]] = [1.0, 0.0, 0.0, 0.0]

                data.mocap_pos[ee2_traj_marker_mids[marker_idx]] = data.mocap_pos[ee2_mid].copy()
                data.mocap_quat[ee2_traj_marker_mids[marker_idx]] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(model, data)

        if i % 100 == 0 or i == N - 1:
            log(
                f"  Render [{i:4d}/{N}]  "
                f"block=({blk_data[i,0]:.3f}, {blk_data[i,1]:.3f}, yaw={blk_data[i,2]:.3f})  "
                f"elapsed={time.time() - t_render:.1f}s"
            )

        renderer.update_scene(data, camera=cam, scene_option=opt)
        frames.append(renderer.render().copy())

    renderer.close()
    log(f"Render done in {time.time() - t_render:.1f}s")

    # =============================================================================
    # Write MP4
    # =============================================================================

    log(f"Writing video to {OUT_PATH} ...")

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
        f"Video -> {OUT_PATH} "
        f"({len(frames) / FPS:.1f}s | {len(frames)} frames)"
    )

    return OUT_PATH


if __name__ == "__main__":
    for idx in range(32):
        for method in ['SVDRO', 'EMPPI', 'DuSTMPC', 'MPC', 'DRO']:
            render_bimanual(idx, method)