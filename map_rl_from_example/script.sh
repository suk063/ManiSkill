python map_rl_from_example/ppo_map.py \
    --env_id=PickYCBSequential-v1 \
    --control_mode=pd_joint_vel \
    --exp_name=YCB_sequential_xarm6_ppo_map \
    --num_envs=32 \
    --num_eval_envs=20 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=300 \
    --num_eval_steps=300 \
    --gamma=0.95 \
    --learning_rate=3e-4 \
    --capture-video \
    --track \
    --wandb_project_name "PPO-RL-Map" \
    # --checkpoint "save_checkpoint/image_only.pt"