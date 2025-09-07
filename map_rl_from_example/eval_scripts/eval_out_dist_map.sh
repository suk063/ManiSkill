python map_rl_from_example/ppo_map_eval.py \
    --env_id=PickYCBSequential-v1 \
    --control_mode=pd_joint_vel \
    --exp_name=YCB_sequential_xarm6_ppo_map_dino_EVAL_OUT_DIST_MAP \
    --num_eval_envs=100 \
    --num_eval_steps=200 \
    --vision_encoder=dino \
    --use_map \
    --object_num=2 \
    --eval_distribution="out" \
    --checkpoint="runs/YCB_sequential_xarm6_ppo_map_dino_zero_2_object_cond/ckpt_601.pt" \
    --start_condition_map \
    # --env_idx=21