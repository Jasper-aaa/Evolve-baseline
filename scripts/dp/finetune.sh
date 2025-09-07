
# This script is used to finetune diffusion policy on downstream tasks

python train.py --config-name=train_fewshot.yaml \
    task=libero_long_fewshot \
    algo=diffusion_policy \
    exp_name=dp-libero-long-1-shot \
    variant_name=block_32 \
    training.use_tqdm=true \
    training.save_all_checkpoints=false \
    training.use_amp=false \
    training.n_epochs=200 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=32 \
    training.auto_continue=false\
    rollout.enabled=false \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=0 \
    logging.mode=disabled \

# Note1: training.auto_continue will automatically load the latest checkpoint from the previous training stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
# Note2: algo.l1_loss_scale is used to finetune the decoder of the autoencoder.
