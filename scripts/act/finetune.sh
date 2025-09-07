
# This script is used to finetune ACT on downstream tasks

python train.py --config-name=train_fewshot.yaml \
    task=libero_long_fewshot \
    algo=act \
    exp_name=libero_long_ft_oneshot \
    variant_name=block_16 \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=16 \
    training.auto_continue=false \
    rollout.enabled=false \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0 \
    logging.mode=disabled \


# Note1: training.auto_continue will automatically load the latest checkpoint from the previous training stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
