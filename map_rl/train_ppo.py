import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import open_clip

# ManiSkill specific imports
import mani_skill
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from model.agent import Agent
from utils.utils import DictArray, GridSampler, Logger, build_checkpoint
from utils.mapping import update_map_online

# Mapping-related imports
from mapping.mapping_lib.voxel_hash_table import VoxelHashTable
from mapping.mapping_lib.implicit_decoder import ImplicitDecoder
from mapping.mapping_lib.visualization import visualize_decoded_features_pca
from mapping.mapping_lib.utils import get_visual_features, get_3d_coordinates


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    wandb_tags: List[str] = field(default_factory=list)
    """additional tags for the wandb run"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "PickCubeDiscreteInit-v1"
    """the id of the environment"""
    robot_uids: str = "panda"
    """the uid of the robot to use in the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 400
    """the number of parallel environments"""
    num_eval_envs: int = 20
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""

    # KL penalty (adaptive)
    use_kl_penalty: bool = True
    """if toggled, add KL penalty term to the policy loss and adapt its coef."""
    kl_coef: float = 1.0
    """initial coefficient for KL penalty term"""
    kl_target: float = 0.01
    """target KL magnitude for adaptive adjustment of kl_coef"""
    kl_adapt_rate: float = 2.0
    """multiplicative factor to adjust kl_coef when KL is above/below target"""
    kl_coef_min: float = 1e-4
    """lower bound for kl_coef when adapting"""
    kl_coef_max: float = 10.0
    """upper bound for kl_coef when adapting"""
    # Dual-Clip PPO
    dual_clip: Optional[float] = 2.0
    """if set, enable Dual-Clip with this coefficient (e.g., 2.0)."""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # task specification
    # (NOTE): the order should match the order of the model_ids in the env
    model_ids: List[str] = field(default_factory=lambda: ["tomato_soup_can", "gelatin_box", "bowl", "apple", "banana"])

    # Environment discretisation
    grid_dim: int = 10
    """Number of cells per axis used for discrete initialisation (N×N grid)."""

    # Map-related arguments
    use_map: bool = False
    """if toggled, use the pre-trained environment map features as part of the observation"""
    use_local_fusion: bool = False
    """if toggled, use the local fusion of the image and map features"""
    vision_encoder: str = "dino" # "plain_cnn" or "dino"
    """the vision encoder to use for the agent"""
    map_dir: str = "mapping/multi_env_maps_cube"
    """Directory where the trained environment maps are stored."""
    decoder_path: str = "mapping/multi_env_maps_cube/shared_decoder.pt"
    """Path to the trained shared decoder model."""

    # Online mapping arguments
    use_online_mapping: bool = False
    """if toggled, update the map online based on robot observations"""
    online_map_update_steps: int = 20
    """the number of optimization steps for online map update per observation"""
    online_decoder_update_steps: int = 5
    """the number of optimization steps for online decoder update per observation"""
    online_map_lr: float = 1e-3
    """the learning rate for the online map optimizer"""
    robot_segmentation_id: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    mapping_step_limit: int = 20
    """the step limit for mapping"""
    map_update_freq: int = 5
    """the frequency of online map updates"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # --- Load CLIP model for text and online mapping ---
    print("--- Loading CLIP model ---")
    clip_model_name = "EVA02-L-14"
    clip_weights_id = "merged2b_s4b_b131k"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_weights_id
    )
    clip_model = clip_model.to(device)
    
    # --- Generate text embeddings for model_ids ---
    text_input = [f"pick up the {s.replace('_', ' ')}" for s in args.model_ids]
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    text_tokens = tokenizer(text_input).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
    print("--- Text embeddings generated ---")

    if args.use_online_mapping:
        print("--- Setting CLIP model for online mapping ---")
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
    else:
        del clip_model, tokenizer
        clip_model = None
        print("--- CLIP model removed from memory ---")

    # --- Load Maps and Decoder ---
    # Always define total number of discrete environments
    total_envs = args.grid_dim ** 2
    # Placeholders for map-related variables to avoid NameError when use_map is False
    all_grids = None
    grid_sampler = None
    active_indices = None
    eval_indices = None
    grids, decoder = None, None
    initial_decoder_state_dict = None
    if args.use_map:
        print("--- Loading maps and decoder for PPO training ---")
        # 1. Load Decoder
        try:
            decoder = ImplicitDecoder(
                voxel_feature_dim=64 * 2, # GRID_FEAT_DIM * GRID_LVLS from map_multi_table.py
                hidden_dim=240,
                output_dim=768,
            ).to(device)
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
            if args.use_online_mapping:
                decoder.train()
                for param in decoder.parameters():
                    param.requires_grad = True
                initial_decoder_state_dict = {k: v.clone() for k, v in decoder.state_dict().items()}
                print(f"Loaded shared decoder from {args.decoder_path} for online training.")
            else:
                decoder.eval()
                for param in decoder.parameters():
                    param.requires_grad = False
                print(f"Loaded shared decoder from {args.decoder_path}")
        except FileNotFoundError:
            print(f"[ERROR] Decoder file not found at {args.decoder_path}. Exiting.")
            exit()

        # 2. Load ALL grids (grid_dim²) and build sampling helper
        total_envs = args.grid_dim ** 2
        all_grids = []
        if not os.path.exists(args.map_dir):
            print(f"[ERROR] Map directory not found: {args.map_dir}. Exiting.")
            exit()

        print(f"Loading {total_envs} maps from {args.map_dir} ...")
        for i in range(total_envs):
            grid_path = os.path.join(args.map_dir, f"env_{i:03d}_grid.pt")
            if not os.path.exists(grid_path):
                print(f"[ERROR] Map file not found: {grid_path}. Exiting.")
                exit()
            all_grids.append(VoxelHashTable.load_dense(grid_path, device=device))
        print(f"--- Loaded {len(all_grids)} maps. ---")

        # Freeze any grid parameters to avoid accidental training
        for grid in all_grids:
            for p in grid.parameters():
                p.requires_grad = False
        # (sampler is created below in a unified way)

    # Helper for a fixed random subset for evaluation (always available)
    def _random_sample_eval(n_envs: int):
        idx = np.random.choice(total_envs, n_envs, replace=False)
        return idx

    # Create GridSampler once. When not using map, create dummy list of length total_envs
    batch_train_envs = args.num_envs if not args.evaluate else 1
    if args.use_map:
        grid_sampler = GridSampler(all_grids, batch_train_envs)
    else:
        # Create a placeholder list to satisfy GridSampler API; we only need indices
        dummy_list = list(range(total_envs))
        grid_sampler = GridSampler(dummy_list, batch_train_envs)

    # Initial training/eval subsets
    active_indices, grids = grid_sampler.sample()
    if not args.use_map:
        grids = None
    eval_indices = _random_sample_eval(args.num_eval_envs)
    
        
    # debug: Visualize the first map
    # if grids and decoder:
    #     print("--- Visualizing map features for the first environment ---")
    #     visualize_decoded_features_pca(grids[0], decoder, device=device)
    #     print("--- Visualization done. Continuing with training/evaluation. ---")

    # env setup
    # env_kwargs = dict(robot_uids=args.robot_uids, obs_mode="rgb+depth+segmentation", render_mode=args.render_mode, sim_backend="physx_cuda", grid_dim=args.grid_dim)
    env_kwargs = dict(robot_uids=args.robot_uids, obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda", grid_dim=args.grid_dim)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=True, state=args.include_state, include_camera_params=True, include_segmentation=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=True, state=args.include_state, include_camera_params=True, include_segmentation=True)

    # visalize segmentation id map
    # Save segmentation id map to file
    segmentation_map = dict(envs.unwrapped.segmentation_id_map.items())
    with open(f"segmentation_id_map.txt", "w") as f:
        for key, value in segmentation_map.items():
            f.write(f"{key}: {value}\n")
    print(f"Segmentation ID map saved to runs/{run_name}/segmentation_id_map.txt")

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            if args.checkpoint:
                eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=list(args.wandb_tags)
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    env_target_obj_idxs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed, options={'global_idx': active_indices.tolist()})
    next_env_target_obj_idx = infos['env_target_obj_idx']
    if args.use_map:
        eval_grids = [all_grids[i] for i in eval_indices]
    else:
        eval_grids = None
    eval_obs, eval_infos = eval_envs.reset(seed=args.seed, options={'global_idx': eval_indices.tolist()})
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(
        envs, 
        sample_obs=next_obs, 
        decoder=decoder if args.use_map else None, 
        use_local_fusion=args.use_local_fusion, 
        vision_encoder=args.vision_encoder,
        text_embeddings=text_embeddings,
    ).to(device)
    # Use differential learning rates for DINO backbone vs. the rest
    if args.vision_encoder == "dino":
        try:
            dino_backbone_params = list(agent.feature_net.vision_encoder.backbone.parameters())
            dino_backbone_param_ids = set(map(id, dino_backbone_params))
            other_params = [p for p in agent.parameters() if id(p) not in dino_backbone_param_ids]
            optimizer = optim.AdamW(
                [
                    {"params": dino_backbone_params, "lr": 1e-5},
                    {"params": other_params, "lr": args.learning_rate},
                ],
                eps=1e-5,
            )
        except AttributeError:
            # Fallback in case the backbone structure changes
            optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # KL penalty coefficient (adaptive)
    kl_coef = args.kl_coef

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        # --------------------------------------------------------------
        # Resample subset of environments for this epoch (train)
        # --------------------------------------------------------------
        # Resample next training subset using GridSampler
        active_indices, grids = grid_sampler.sample()
        if not args.use_map:
            grids = None
        else:
            # For online mapping, create a copy of the grids for this rollout
            if args.use_online_mapping:
                # if initial_decoder_state_dict:
                #     decoder.load_state_dict(initial_decoder_state_dict)
                online_grids = [grid.clone() for grid in grids]
                for grid in online_grids:
                    for p in grid.parameters():
                        p.requires_grad = True
                # Create a single optimizer for all map and decoder parameters
                all_map_params = [p for grid in online_grids for p in grid.parameters()]
                map_optimizer = optim.Adam(all_map_params + list(decoder.parameters()), lr=args.online_map_lr)
            else:
                online_grids = grids
                map_optimizer = None

        next_obs, infos = envs.reset(options={'global_idx': active_indices.tolist()})

        next_done = torch.zeros(args.num_envs, device=device)
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            # Use FIXED evaluation subset (eval_indices, eval_grids) sampled once at start
            eval_obs, eval_infos = eval_envs.reset(options={'global_idx': eval_indices.tolist()})

            # For online mapping during evaluation
            if args.use_online_mapping and args.use_map:
                # if initial_decoder_state_dict:
                #     decoder.load_state_dict(initial_decoder_state_dict)
                online_eval_grids = [grid.clone() for grid in eval_grids]
                for grid in online_eval_grids:
                    for p in grid.parameters():
                        p.requires_grad = True
                all_eval_map_params = [p for grid in online_eval_grids for p in grid.parameters()]
                eval_map_optimizer = optim.Adam(all_eval_map_params + list(decoder.parameters()), lr=args.online_map_lr)
            else:
                online_eval_grids = eval_grids
                eval_map_optimizer = None

            # (NOTE): debugging code to save target objects ---
            target_obj_indices = eval_infos['env_target_obj_idx'].cpu().numpy()
            target_obj_names = [args.model_ids[i] for i in target_obj_indices]
            output_path = os.path.join(f"runs/{run_name}", "eval_target_objects.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                for i, name in enumerate(target_obj_names):
                    f.write(f"eval_env_{i}: {name}\n")
            print(f"Saved eval target objects to {output_path}")
            # --- End of debugging code ---

            eval_metrics = defaultdict(list)
            num_episodes = 0
            for step in range(args.num_eval_steps):
                with torch.no_grad():
                    action = agent.get_action(eval_obs, map_features=online_eval_grids if args.use_map else None, env_target_obj_idx=eval_infos['env_target_obj_idx'], deterministic=True)
                
                eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(action)

                # Online map update for evaluation
                if args.use_online_mapping:
                    if step % args.map_update_freq == 0:
                        is_obj_grasped = eval_infos['is_obj_grasped']
                        update_mask = ~is_obj_grasped
                        update_map_online(eval_obs, eval_obs['sensor_param'], online_eval_grids, clip_model, decoder, eval_map_optimizer, args, update_mask=update_mask)

                # eval_infos['env_target_obj_idx']

                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            # torch.save(agent.state_dict(), model_path)
            torch.save(build_checkpoint(agent, args, envs), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            env_target_obj_idxs[step] = next_env_target_obj_idx

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, map_features=online_grids if args.use_online_mapping else grids if args.use_map else None, env_target_obj_idx=next_env_target_obj_idx)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)

            # infos['env_target_obj_idx']
            next_env_target_obj_idx = infos['env_target_obj_idx']
            
            # Online map update
            if args.use_online_mapping:
                if step % args.map_update_freq == 0:
                    is_obj_grasped = infos['is_obj_grasped']
                    update_mask = ~is_obj_grasped
                    update_map_online(next_obs, next_obs['sensor_param'], online_grids, clip_model, decoder, map_optimizer, args, update_mask=update_mask)

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                # Slice final observations for environments that terminated
                final_obs = infos["final_observation"]
                def _slice(o):
                    if isinstance(o, dict):
                        return {kk: _slice(vv) for kk, vv in o.items()}
                    elif torch.is_tensor(o):
                        return o[done_mask]
                    else:
                        # assume numpy array or list convertible to tensor indexing
                        return o[done_mask]
                final_obs = _slice(final_obs)
                with torch.no_grad():
                    idx = torch.arange(args.num_envs, device=device)[done_mask]
                    if len(idx) > 0:
                        final_grids = [grid for i, grid in enumerate(online_grids if args.use_online_mapping else grids) if done_mask[i]] if args.use_map else None
                        final_values[step, idx] = agent.get_value(final_obs, map_features=final_grids, env_target_obj_idx=final_info['env_target_obj_idx'][done_mask]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs, map_features=online_grids if args.use_online_mapping else grids if args.use_map else None, env_target_obj_idx=next_env_target_obj_idx).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_env_target_obj_idxs = env_target_obj_idxs.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_grids = [grids[i % args.num_envs] for i in mb_inds] if args.use_map else None
                if args.use_online_mapping:
                    mb_grids = [online_grids[i % args.num_envs] for i in mb_inds]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], map_features=mb_grids, env_target_obj_idx=b_env_target_obj_idxs[mb_inds], action=b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # differentiable KL approximation for penalty term
                approx_kl_loss = ((ratio - 1) - logratio).mean()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (with Dual-Clip option)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_unreduced = torch.max(pg_loss1, pg_loss2)
                if args.dual_clip is not None:
                    neg_mask = mb_advantages < 0
                    dual_bound = -args.dual_clip * mb_advantages
                    pg_loss_unreduced = torch.where(neg_mask, torch.max(pg_loss_unreduced, dual_bound), pg_loss_unreduced)
                pg_loss = pg_loss_unreduced.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # Add KL penalty if enabled
                kl_term = kl_coef * approx_kl_loss if args.use_kl_penalty else 0.0
                loss = pg_loss + kl_term - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        # Adapt KL coefficient once per epoch, using the measured approx_kl
        if args.use_kl_penalty:
            try:
                kl_val = float(approx_kl.item())
                if kl_val > args.kl_target:
                    kl_coef = min(kl_coef * args.kl_adapt_rate, args.kl_coef_max)
                else:
                    kl_coef = max(kl_coef / args.kl_adapt_rate, args.kl_coef_min)
            except NameError:
                pass
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.use_kl_penalty:
            logger.add_scalar("losses/kl_coef", kl_coef, global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        # torch.save(agent.state_dict(), model_path)
        torch.save(build_checkpoint(agent, args, envs), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    eval_envs.close()
    if logger is not None: logger.close()
