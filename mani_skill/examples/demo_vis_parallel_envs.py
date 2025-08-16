import signal
import sys

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work

import torch

from mani_skill.utils import common
from mani_skill.utils import visualization
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickYCBCustom-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Can be rgb or rgb+depth, rgb+normal, albedo+depth etc. Which ever image-like textures you want to visualize can be tacked on")
    parser.add_argument("-r", "--robot", type=str, default="xarm6_robotiq", help="The robot to use")
    parser.add_argument("-ne", "--num-eps", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("-n", "--num-envs", type=int, default=4, help="Number of parallel environments to run and visualize")
    parser.add_argument("-cw", "--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("-ch", "--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument("-c", "--camera", type=str, help="Specific camera to visualize. If not specified, will show all cameras")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

import numpy as np


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    sensor_configs["shader_pack"] = args.shader
    
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        robot_uids=args.robot,
        sensor_configs=sensor_configs
    )

    obs, _ = env.reset(seed=args.seed)
    
    # Print available cameras
    available_cameras = []
    for cam_name, config in env.unwrapped._sensors.items():
        if isinstance(config, Camera):
            available_cameras.append(cam_name)
    
    print(f"Available cameras: {available_cameras}")
    print(f"Number of parallel environments: {args.num_envs}")
    print(f"Observation mode: {args.obs_mode}")
    print(f"Robot: {args.robot}")
    print(f"Shader: {args.shader}")
    
    # Filter cameras if specific camera is requested
    cameras_to_show = available_cameras
    if args.camera:
        if args.camera in available_cameras:
            cameras_to_show = [args.camera]
            print(f"Visualizing only camera: {args.camera}")
        else:
            print(f"Warning: Camera '{args.camera}' not found. Available cameras: {available_cameras}")
            print("Showing all cameras instead.")
    
    renderer = visualization.ImageRenderer()

    for episode in range(args.num_eps):
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            cam_num = 0
            imgs = []
            for env_idx in range(args.num_envs):
                for cam in cameras_to_show:
                    if cam in obs["sensor_data"]:
                        for texture in obs["sensor_data"][cam].keys():
                            if obs["sensor_data"][cam][texture].dtype == torch.uint8:
                                data = common.to_numpy(obs["sensor_data"][cam][texture][env_idx])
                                imgs.append(data)
                            else:
                                data = common.to_numpy(obs["sensor_data"][cam][texture][env_idx]).astype(np.float32)
                                # Handle potential division by zero
                                data_range = data.max() - data.min()
                                if data_range > 0:
                                    data = data / data_range
                                else:
                                    data = data - data.min()  # Just normalize to [0, 1]
                                data_rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
                                data_rgb[..., :] = data * 255
                                imgs.append(data_rgb)
                cam_num += 1
            img = visualization.tile_images(imgs, nrows=int(np.sqrt(args.num_envs)))
            renderer(img)
            if terminated.any() or truncated.any():
                print(f"Episode {episode+1}/{args.num_eps} finished")
                obs, _ = env.reset(seed=args.seed)
                break
    
if __name__ == "__main__":
    main(parse_args()) 