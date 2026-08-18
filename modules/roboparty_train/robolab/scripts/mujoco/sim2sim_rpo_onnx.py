# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import mujoco
import onnxruntime as ort
import os
from pathlib import Path

try:
    from .rpo_21_mujoco import (
        RPO_MJCF_JOINT_NAMES,
        RPO_TAU_LIMIT,
        assert_rpo_21_mujoco_model,
        get_obs,
    )
except ImportError:
    from rpo_21_mujoco import (
        RPO_MJCF_JOINT_NAMES,
        RPO_TAU_LIMIT,
        assert_rpo_21_mujoco_model,
        get_obs,
    )

from robolab.assets import ISAAC_DATA_DIR

try:
    import mujoco_viewer
except ImportError:
    mujoco_viewer = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


LOCOMOTION_POLICY_JOINT_NAMES = [
    "head_yaw_joint",
    "left_leg_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_leg_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_leg_roll_joint",
    "left_shoulder_roll_joint",
    "right_leg_roll_joint",
    "right_shoulder_roll_joint",
    "left_leg_yaw_joint",
    "left_shoulder_yaw_joint",
    "right_leg_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_knee_joint",
    "left_elbow_pitch_joint",
    "right_knee_joint",
    "right_elbow_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
LOCOMOTION_POLICY_TO_MJCF = [
    RPO_MJCF_JOINT_NAMES.index(name) for name in LOCOMOTION_POLICY_JOINT_NAMES
]
LOCOMOTION_DEFAULT_POS_POLICY = np.array(
    [
        0.0, -0.15, -0.18, -0.15, -0.18, 0.0, 0.25,
        0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.25,
        -0.3, 0.25, -0.3, -0.1, -0.1, 0.0, 0.0,
    ],
    dtype=np.double,
)
LOCOMOTION_POLICY_SIGNS = np.array(
    [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
    dtype=np.double,
)
LOCOMOTION_DEFAULT_POS_MJCF = np.zeros(len(LOCOMOTION_POLICY_JOINT_NAMES), dtype=np.double)
LOCOMOTION_DEFAULT_POS_MJCF[LOCOMOTION_POLICY_TO_MJCF] = (
    LOCOMOTION_POLICY_SIGNS * LOCOMOTION_DEFAULT_POS_POLICY
)
LOCOMOTION_KPS_MJCF = np.array(
    [
        100.0, 100.0, 100.0, 150.0, 60.0, 60.0,
        100.0, 100.0, 100.0, 150.0, 60.0, 60.0,
        30.0,
        40.0, 40.0, 40.0, 30.0,
        40.0, 40.0, 40.0, 30.0,
    ],
    dtype=np.double,
)
LOCOMOTION_KDS_MJCF = np.array(
    [
        3.3, 3.3, 3.3, 4.0, 2.5, 2.5,
        3.3, 3.3, 3.3, 4.0, 2.5, 2.5,
        1.5,
        2.0, 2.0, 2.0, 1.5,
        2.0, 2.0, 2.0, 1.5,
    ],
    dtype=np.double,
)


class cmd:
    vx = 0.7
    vy = 0.0
    dyaw = 0.0


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg, headless=False, no_video=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    assert_rpo_21_mujoco_model(model, cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)
    
    #os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    # 根据 headless 参数选择渲染模式
    renderer = None
    out = None
    cam = None
    viewer = None
    if headless:
        if not no_video:
            import cv2

            renderer = mujoco.Renderer(model, width=1920, height=1080)
            # 设置视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 创建并配置相机
            cam = mujoco.MjvCamera()
            cam.distance = 4.0      # 增加距离以获得更好的视角
            cam.azimuth = 45.0     # 水平旋转角度
            cam.elevation = -20.0   # 垂直俯仰角度
            cam.lookat = [0, 0, 1]  # 观察点位置
            out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        if mujoco_viewer is None:
            raise RuntimeError("mujoco_viewer is required when running without --headless.")
        mode = 'window'
        viewer = mujoco_viewer.MujocoViewer(model, data, mode=mode, width=1920, height=1080)
        # 设置窗口模式下的相机参数
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs), dtype=np.double)
    hist_obs.fill(0.0)

    count_lowlevel = 0

    # --- Data collection lists for plotting (LOW FREQUENCY ONLY) ---
    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)  # Initialize tau
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = [] # Store [vx, vy] at low freq
    actual_ang_vel_data = [] # Store [wz] at low freq
    # -------------------------------------------------------------
    min_base_height = float("inf")
    max_gravity_z = -float("inf")
    final_gravity = np.array([0.0, 0.0, -1.0], dtype=np.double)
    is_first_frame = True
    for step in tqdm(
        range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
        desc="Simulating...",
        disable=headless,
    ):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        min_base_height = min(min_base_height, float(q[2]))
        max_gravity_z = max(max_gravity_z, float(gvec[2]))
        final_gravity = gvec.copy()
        q = q[-cfg.robot_config.num_actions:]
        dq = dq[-cfg.robot_config.num_actions:]

        # 1000hz -> 100hz/50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            q_ = q - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = cfg.robot_config.policy_joint_signs[i] * q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = cfg.robot_config.policy_joint_signs[i] * dq[cfg.robot_config.usd2urdf[i]]

            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            
            obs[0, 0:3] = omega
            obs[0, 3:6] = gvec
            obs[0, 6] = cmd.vx 
            obs[0, 7] = cmd.vy 
            obs[0, 8] = cmd.dyaw 
            num_actions = cfg.robot_config.num_actions
            obs[0, 9 : 9 + num_actions] = q_obs
            obs[0, 9 + num_actions : 9 + 2 * num_actions] = dq_obs
            obs[0, 9 + 2 * num_actions : 9 + 3 * num_actions] = action

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            if cfg.robot_config.obs_stack_order == "frame_major":
                stacked_obs = hist_obs.reshape(1, -1)
            else:
                stacked_fields = []
                field_offset = 0
                for field_size in cfg.robot_config.obs_field_sizes:
                    field_end = field_offset + field_size
                    stacked_fields.append(hist_obs[:, field_offset:field_end].reshape(-1))
                    field_offset = field_end
                if field_offset != cfg.robot_config.num_single_obs:
                    raise ValueError(
                        f"Observation fields total {field_offset}, expected "
                        f"{cfg.robot_config.num_single_obs}."
                    )
                stacked_obs = np.concatenate(stacked_fields).reshape(1, -1)

            policy_input = np.clip(
                stacked_obs,
                -cfg.robot_config.clip_observations,
                cfg.robot_config.clip_observations,
            ).astype(np.float32)
            if policy_input.shape[1] != cfg.robot_config.num_observations:
                raise ValueError(
                    f"Policy input has {policy_input.shape[1]} observations, "
                    f"expected {cfg.robot_config.num_observations}."
                )
            policy_action = policy(policy_input).reshape(-1)
            if policy_action.shape[0] != cfg.robot_config.num_actions:
                raise ValueError(
                    f"Policy output has {policy_action.shape[0]} actions, "
                    f"expected {cfg.robot_config.num_actions}."
                )
            action[:] = policy_action

            target_q = np.clip(
                action, -cfg.robot_config.clip_actions, cfg.robot_config.clip_actions
            ) * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = (
                    cfg.robot_config.policy_joint_signs[i] * target_q[i]
                )
            target_pos = target_pos + cfg.robot_config.default_pos
            joint_ranges = model.jnt_range[-cfg.robot_config.num_actions :]
            target_pos = np.clip(
                target_pos,
                joint_ranges[:, 0] + cfg.robot_config.joint_limit_margin,
                joint_ranges[:, 1] - cfg.robot_config.joint_limit_margin,
            )

            # --- Capture actual state at this low-frequency step ---
            # Note: q, v, omega were just computed by get_obs() for the current simulation step
            q_low_freq = q.copy()
            v_low_freq = v[:2].copy() # Capture x and y linear velocity
            omega_low_freq = omega[2].copy() # Capture z angular velocity
            # -----------------------------------------------------

            # --- Collect low-frequency data for plotting ---
            # Use the exact simulation time at this low-freq step
            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq) # Use the captured actual joint pos
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v_low_freq) # Use the captured actual lin vel
            actual_ang_vel_data.append(omega_low_freq) # Use the captured actual ang vel
            # ----------------------------------------------

            if headless:
                if no_video:
                    pass
                else:
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render()  # 直接获取RGB图像
                    out.write(img)
            else:
                viewer.render()
            
        target_vel = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_pos, q, cfg.robot_config.kps,
                        target_vel, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau
        mujoco.mj_step(model, data)

        count_lowlevel += 1

    if headless:
        if not no_video:
            out.release()
    else:
        viewer.close()

    print(
        "Simulation summary: "
        f"min_base_height={min_base_height:.6f}, "
        f"final_base_height={float(data.qpos[2]):.6f}, "
        f"max_gravity_z={max_gravity_z:.6f}, "
        f"final_gravity=[{final_gravity[0]:.6f}, {final_gravity[1]:.6f}, {final_gravity[2]:.6f}]"
    )

     # --- Plotting Section (Using only low-frequency data) ---

    if cfg.sim_config.save_plots:
        import matplotlib.pyplot as plt

        print("Simulation finished. Generating plots...")

        # Convert collected data to numpy arrays
        time_data = np.array(time_data)
        commanded_joint_pos_data = np.array(commanded_joint_pos_data)
        actual_joint_pos_data = np.array(actual_joint_pos_data)
        tau_data = np.array(tau_data)
        commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
        commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
        commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
        actual_lin_vel_data = np.array(actual_lin_vel_data)
        actual_ang_vel_data = np.array(actual_ang_vel_data)


        # Plot 1: Commanded vs Actual Joint Positions
        num_joints = cfg.robot_config.num_actions
        n_cols = 4 # Or adjust based on num_joints
        n_rows = (num_joints + n_cols - 1) // n_cols

        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        axes1 = axes1.flatten()

        joint_names = [f'Joint {i+1}' for i in range(num_joints)] # Generic names (consider using specific robot joint names if available)

        for i in range(num_joints):
            ax = axes1[i]
            # Plotting low-frequency commanded and actual joint positions
            ax.plot(time_data, commanded_joint_pos_data[:, i], label='Commanded', linestyle='--')
            ax.plot(time_data, actual_joint_pos_data[:, i], label='Actual')
            ax.set_title(joint_names[i])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position [rad]")
            ax.legend()
            ax.grid(True)

        # Hide any unused subplots
        for i in range(num_joints, len(axes1)):
            fig1.delaxes(axes1[i])

        fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
        plt.tight_layout()


        # Plot 2: Commanded vs Actual Base Velocities
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Linear Velocity X
        # Plotting low-frequency commanded and actual velocities
        axes2[0].plot(time_data, commanded_lin_vel_x_data, label='Commanded Vx', linestyle='--')
        axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label='Actual Vx')
        axes2[0].set_title("Base Linear Velocity X")
        axes2[0].set_xlabel("Time [s]")
        axes2[0].set_ylabel("Velocity [m/s]")
        axes2[0].legend()
        axes2[0].grid(True)

        # Linear Velocity Y
        axes2[1].plot(time_data, commanded_lin_vel_y_data, label='Commanded Vy', linestyle='--')
        axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label='Actual Vy')
        axes2[1].set_title("Base Linear Velocity Y")
        axes2[1].set_xlabel("Time [s]")
        axes2[1].set_ylabel("Velocity [m/s]")
        axes2[1].legend()
        axes2[1].grid(True)

        # Angular Velocity Z
        axes2[2].plot(time_data, commanded_ang_vel_z_data, label='Commanded Dyaw', linestyle='--')
        axes2[2].plot(time_data, actual_ang_vel_data, label='Actual Dyaw') # actual_ang_vel_data is already 1D
        axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
        axes2[2].set_xlabel("Time [s]")
        axes2[2].set_ylabel("Angular Velocity [rad/s]")
        axes2[2].legend()
        axes2[2].grid(True)

        fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
        plt.tight_layout()

        fig1.savefig("joint_positions.png")
        fig2.savefig("base_velocities.png")

        print("Plots finished.")
    # --- End Plotting Section ---

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, help='Path to the ONNX policy.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--headless', action='store_true',
                      help='Run without GUI and save video')
    parser.add_argument('--no-video', action='store_true',
                      help='Run headless without creating a MuJoCo renderer/video')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save joint/base velocity plots after simulation')
    parser.add_argument('--obs-stack-order', choices=('frame_major', 'obs_major'),
                      default='frame_major', help='History layout expected by the ONNX policy')
    args = parser.parse_args()
    model_path = Path(args.load_model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX policy not found: {model_path}")

    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo_21_terrain.xml'
            else:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo_21.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
            save_plots = args.save_plots

        class robot_config:
            mjcf_joint_names = RPO_MJCF_JOINT_NAMES
            kps = LOCOMOTION_KPS_MJCF
            kds = LOCOMOTION_KDS_MJCF
            default_pos = LOCOMOTION_DEFAULT_POS_MJCF
            tau_limit = RPO_TAU_LIMIT
            frame_stack = 10
            num_actions = len(LOCOMOTION_POLICY_JOINT_NAMES)
            num_single_obs = 9 + 3 * num_actions
            num_observations = num_single_obs * frame_stack
            obs_field_sizes = (3, 3, 3, num_actions, num_actions, num_actions)
            obs_stack_order = args.obs_stack_order
            action_scale = 0.25
            clip_observations = 100.0
            clip_actions = 100.0
            usd2urdf = LOCOMOTION_POLICY_TO_MJCF
            policy_joint_signs = LOCOMOTION_POLICY_SIGNS
            joint_limit_margin = 0.0

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if len(input_shape) >= 2 and isinstance(input_shape[1], int) and input_shape[1] != Sim2simCfg.robot_config.num_observations:
        raise ValueError(
            f"{model_path} input dim is {input_shape[1]}, expected "
            f"{Sim2simCfg.robot_config.num_observations} for 21-DoF RPO default sim2sim."
        )
    if len(output_shape) >= 2 and isinstance(output_shape[1], int) and output_shape[1] != Sim2simCfg.robot_config.num_actions:
        raise ValueError(
            f"{model_path} output dim is {output_shape[1]}, expected "
            f"{Sim2simCfg.robot_config.num_actions} for 21-DoF RPO."
        )

    def policy(policy_input):
        return session.run([output_name], {input_name: policy_input.astype("float32")})[0]

    run_mujoco(policy, Sim2simCfg(), args.headless, args.no_video)
