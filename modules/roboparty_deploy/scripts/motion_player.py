#!/usr/bin/env python3
import sys
import time
import logging
import argparse
import numpy as np
import robot_py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

JOINT_NUM = 21
JOINT_DEFAULT_ANGLE = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
    -0.14782176911830902, 0.39768826961517334, 0.04596780240535736, -0.012397955171763897,
    -0.032616160809993744, -0.5540932416915894, 0.00667582219466567, -0.09136339277029037,
])
USD2URDF = [0, 6, 12, 1, 7, 13, 17, 2, 8, 14, 18, 3, 9, 15, 19, 4, 10, 16, 20, 5, 11]

class MotionLoader:
    def __init__(self, motion_file: str, logger, usd2urdf: bool = False):
        try:
            data = np.load(motion_file)
            self.fps = int(data['fps'].item())
            pos_usd = data['joint_pos']
            vel_usd = data['joint_vel']
            if pos_usd.shape[1] != JOINT_NUM or vel_usd.shape[1] != JOINT_NUM:
                raise RuntimeError(
                    f"Motion file must contain {JOINT_NUM} joints, got "
                    f"joint_pos={pos_usd.shape[1]}, joint_vel={vel_usd.shape[1]}"
                )
            self.joint_default_angle = JOINT_DEFAULT_ANGLE.copy()
            self.joint_pos = pos_usd.copy()
            self.joint_vel = vel_usd.copy()
            if usd2urdf:
                logger.info("Converting joint order from USD to URDF")
                self.joint_pos[:, USD2URDF] = pos_usd
                self.joint_vel[:, USD2URDF] = vel_usd
            self.joint_pos -= self.joint_default_angle
            
            self.num_frames = self.joint_pos.shape[0]
            self.num_joints = self.joint_pos.shape[1]
            
            logger.info(f"Loaded motion file: {motion_file}")
            logger.info(f"FPS: {self.fps}, Frames: {self.num_frames}, Joints: {self.num_joints}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load motion file: {e}")
    
    def get_pos(self, frame: int) -> np.ndarray:
        return self.joint_pos[frame]
    
    def get_vel(self, frame: int) -> np.ndarray:
        return self.joint_vel[frame]


class MotionPlayer: 
    def __init__(self, motion_file: str, config_file: str, speed: float = 1.0, usd2urdf: bool = False):
        self.logger = logging.getLogger("MotionPlayer")
        self.motion_loader = MotionLoader(motion_file, self.logger, usd2urdf)
        self.positions = [0.0] * self.motion_loader.num_joints
        
        # 初始化 RobotInterface
        try:
            self.robot = robot_py.RobotInterface(config_file)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize robot interface: {e}")
        
        self.robot.init_motors()
        time.sleep(1)
        self.robot.reset_joints(self.motion_loader.joint_default_angle) 
        time.sleep(1)

        # 播放控制
        self.is_playing = False
        self.speed = min(max(0.1, speed), 1.0)
        self.logger.info(f"Playback speed: {self.speed}x")

        self.step = int(200 / (self.motion_loader.fps * self.speed))
        self.i = 0
        self.is_playing = True
        self.period = 1.0/200


    def update_frame(self, frame_idx: int):
        if frame_idx >= self.motion_loader.num_frames:
            self.logger.warning(f"Frame index {frame_idx} out of range")
            return
        positions = self.motion_loader.get_pos(frame_idx)
        self.positions = positions.tolist()
    
    def step_once(self):
        if self.is_playing:
            if self.i % self.step == 0:
                frame_idx = self.i // self.step
                if frame_idx >= self.motion_loader.num_frames:
                    self.is_playing = False
                    self.logger.info("Motion playback finished")
                    return
                self.update_frame(frame_idx)
            self.robot.apply_action(self.positions)
            self.i += 1
    
    def run(self):
        self.logger.info("Starting playback loop...")
        try:
            while self.is_playing:
                start_time = time.time()
                self.step_once()
                elapsed = time.time() - start_time
                sleep_time = max(0, self.period - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            self.stop()
            self.logger.info("Interrupted by user")

    def stop(self):
        self.is_playing = False

def parse_args():
    parser = argparse.ArgumentParser(description='Motion Player - 播放 motion 数据并发布关节状态')
    parser.add_argument('--motion_file', type=str, required=True, help='Motion 文件路径')
    parser.add_argument('--config_file', type=str, required=True, help='Robot config 文件路径')
    parser.add_argument('--speed', type=float, default=1.0, help='播放速度倍率')
    parser.add_argument('--usd2urdf', action='store_true',  help='是否将 USD 关节顺序转换为 URDF 关节顺序，默认关闭')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    try:
        player = MotionPlayer(args.motion_file, args.config_file, args.speed, args.usd2urdf)
        player.run()
    except Exception as e:
        if 'player' in locals():
            player.logger.error(f"Error: {e}")
        else:
            print(f"Error: {e}")
        return 1
    finally:
        pass
    return 0

if __name__ == '__main__':
    sys.exit(main())
