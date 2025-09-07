
# This script is used to finetune ResNet-T on downstream tasks

python train.py --config-name=train_fewshot.yaml \
    task=libero_long_fewshot \
    algo=bc_transformer \
    exp_name=bctrans-libero-long-1shot \
    variant_name=block_10 \
    training.use_tqdm=true \
    training.save_all_checkpoints=false \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    train_dataloader.batch_size=128 \
    make_unique_experiment_dir=true \
    training.auto_continue=false \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=1 \
    seed=0 \
    logging.mode=disabled \

# Note1: training.auto_continue will automatically load the latest checkpoint from the previous training stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
