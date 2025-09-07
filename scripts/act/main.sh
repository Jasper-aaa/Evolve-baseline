
# This script is used to train the ACT model

python train.py --config-name=train_prior.yaml \
    task=libero_90 \
    algo=act \
    exp_name=pretrain_act_libero_90 \
    variant_name=block_16 \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=16 \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0 \
    checkpoint_path="/home/yeyifan/workplace/embodied/experiments/libero/LIBERO_90/act/pretrain_act_libero_90/block_16/0/run_000/multitask_model_epoch_0070.pth" \

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
