python map_rl_from_example/ppo_map.py \
    --env_id=PickYCBSequential-v1 \
    --control_mode=pd_joint_vel \
    --exp_name=YCB_sequential_xarm6_ppo_map_dino_zero_2_object_2e-3 \
    --num_envs=100 \
    --num_eval_envs=20 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=200 \
    --num_eval_steps=200 \
    --gamma=0.9 \
    --ent_coef=2e-3 \
    --learning_rate=3e-4 \
    --vision_encoder=dino \
    --capture-video \
    --track \
    --wandb_project_name "PPO-RL-Map" \
    --use_map \
    --object_num=2

    # --start_condition_map
