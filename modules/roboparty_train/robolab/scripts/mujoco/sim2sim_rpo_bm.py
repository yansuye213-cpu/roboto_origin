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

# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import numpy as np
import mujoco
from tqdm import tqdm
try:
    from .rpo_21_mujoco import (
        RPO_ACTION_JOINT_NAMES,
        RPO_ACTION_TO_MJCF,
        RPO_DEFAULT_POS,
        RPO_KDS,
        RPO_KPS,
        RPO_TAU_LIMIT,
        assert_rpo_21_mujoco_model,
        get_obs as get_rpo_obs,
    )
except ImportError:
    from rpo_21_mujoco import (
        RPO_ACTION_JOINT_NAMES,
        RPO_ACTION_TO_MJCF,
        RPO_DEFAULT_POS,
        RPO_KDS,
        RPO_KPS,
        RPO_TAU_LIMIT,
        assert_rpo_21_mujoco_model,
        get_obs as get_rpo_obs,
    )
from robolab.assets import ISAAC_DATA_DIR
import time

try:
    import mujoco_viewer
except ImportError:
    mujoco_viewer = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    from loop_rate_limiters import RateLimiter
except ImportError:
    class RateLimiter:
        def __init__(self, frequency, warn=False):
            self.period = 1.0 / float(frequency)

        def sleep(self):
            time.sleep(self.period)


class cmd:
    camera_follow = True
    reset_requested = False

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")
    
    @classmethod
    def reset(cls):
        print(f"Reset")

def on_press(key):
    try:
        if key.char == 'f':
            cmd.toggle_camera_follow()
        elif key.char == '0':
            cmd.reset_requested = True
    except AttributeError:
        pass

def on_release(key):
    pass

def start_keyboard_listener():
    if keyboard is None:
        print("[WARN] pynput is not installed; keyboard controls are disabled.")
        return None
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def get_obs(data):
    return get_rpo_obs(data)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _require(module, package_name, purpose):
    if module is None:
        raise RuntimeError(f"{package_name} is required {purpose}.")
    return module

def run_mujoco(policy, cfg, headless=False,loop=False,motion_file=None,no_video=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1
    
    print("=" * 60)
    print("Keyboard control instructions:")
    print("  0 key: Reset all speeds to 0")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    if motion_file is None:
        raise ValueError("--motion_file is required for BM sim2sim.")
    motion=np.load(motion_file)
    motion_pos=motion["body_pos_w"]
    motion_quat=motion["body_quat_w"]
    m_input_pos=motion["joint_pos"]
    m_input_vel=motion["joint_vel"]
    if m_input_pos.shape[1] != cfg.robot_config.num_actions or m_input_vel.shape[1] != cfg.robot_config.num_actions:
        raise ValueError(
            f"{motion_file} has joint_pos/joint_vel dims "
            f"{m_input_pos.shape[1]}/{m_input_vel.shape[1]}, expected {cfg.robot_config.num_actions}. "
            "Regenerate BM motion data with the 21-DoF RPO asset."
        )

    num_frames = min(m_input_pos.shape[0], m_input_vel.shape[0], motion_pos.shape[0], motion_quat.shape[0])


    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    assert_rpo_21_mujoco_model(model, cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    data.qpos[0:3] = motion_pos[0,0,:]
    data.qpos[3:7] = motion_quat[0,0,:]
    mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    if headless:
        if not no_video:
            cv2_mod = _require(cv2, "opencv-python", "when running headless video output")
            renderer = mujoco.Renderer(model, width=1920, height=1080)
            fourcc = cv2_mod.VideoWriter_fourcc(*'mp4v')
            cam = mujoco.MjvCamera()
            cam.distance = 4.0      # 增加距离以获得更好的视角
            cam.azimuth = 45.0     # 水平旋转角度
            cam.elevation = -20.0   # 垂直俯仰角度
            cam.lookat = [0, 0, 1]  # 观察点位置
            out = cv2_mod.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        viewer_mod = _require(mujoco_viewer, "mujoco-python-viewer", "when running with the GUI viewer")
        mode = 'window'
        viewer = viewer_mod.MujocoViewer(model, data, mode=mode, width=1920, height=1080)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs), dtype=np.double)
    hist_obs.fill(0.0)

    count_lowlevel = 0
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    is_first_frame = True

    control_freq = 1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation)
    motion_t=0

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            cmd.reset()
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False
            motion_t = 0
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot_config.num_actions:]
        dq = dq[-cfg.robot_config.num_actions:]

        # 1000hz -> 100hz/50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            idx=frame_idx(motion_t)

            m_input=np.concatenate((m_input_pos[idx,:],m_input_vel[idx,:]),axis=0)
            
            q_ = q - cfg.robot_config.default_pos

            q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            
            num_actions = cfg.robot_config.num_actions
            motion_dim = 2 * num_actions
            obs[0, 0:motion_dim] = m_input
            obs[0, motion_dim : motion_dim + 3] = omega
            obs[0, motion_dim + 3 : motion_dim + 6] = gvec
            obs[0, motion_dim + 6 : motion_dim + 6 + num_actions] = q_obs
            obs[0, motion_dim + 6 + num_actions : motion_dim + 6 + 2 * num_actions] = dq_obs
            obs[0, motion_dim + 6 + 2 * num_actions : motion_dim + 6 + 3 * num_actions] = action

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            if policy_input.shape[1] != cfg.robot_config.num_observations:
                raise ValueError(
                    f"Policy input has {policy_input.shape[1]} observations, "
                    f"expected {cfg.robot_config.num_observations}."
                )
            with torch.inference_mode():
                policy_action = policy(torch.tensor(policy_input))[0].detach().numpy()
            if policy_action.shape[0] != cfg.robot_config.num_actions:
                raise ValueError(
                    f"Policy output has {policy_action.shape[0]} actions, "
                    f"expected {cfg.robot_config.num_actions}."
                )
            action[:] = policy_action

            target_q = action * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]
            target_pos = target_pos + cfg.robot_config.default_pos

            if headless and not no_video:
                renderer.update_scene(data, camera=cam)
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                img = renderer.render() 
                out.write(img)
            else:
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()

            motion_t+=1
            rate_limiter = RateLimiter(frequency=control_freq, warn=False)
            rate_limiter.sleep()
            
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
    if keyboard_listener is not None:
        keyboard_listener.stop()
    print("Simulation finished. Generating plots...")


    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--headless', action='store_true',
                      help='Run without GUI and save video')
    parser.add_argument('--no-video', action='store_true',
                      help='Run headless without creating a MuJoCo renderer/video')
    parser.add_argument('--motion_file',type=str,help='path to motion file(npz)')
    parser.add_argument('--loop',action="store_true",help='loop the policy')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo_21_terrain.xml'
            else:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo_21.xml'
            sim_duration = 1000.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = RPO_KPS
            kds = RPO_KDS
            default_pos = RPO_DEFAULT_POS
            tau_limit = RPO_TAU_LIMIT
            frame_stack = 1
            num_actions = len(RPO_ACTION_JOINT_NAMES)
            num_single_obs = 6 + 5 * num_actions
            num_observations = num_single_obs * frame_stack
            action_scale = 0.25
            usd2urdf = RPO_ACTION_TO_MJCF

    if torch is None:
        raise RuntimeError("torch is required to load a TorchScript policy.")
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless,args.loop,args.motion_file,args.no_video)
