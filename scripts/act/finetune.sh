task="$1"
# This script is used to finetune ACT on downstream tasks
for shot in 1 2 4 8
do
    python train.py --config-name=train_fewshot.yaml \
        task=libero_${task}_fewshot \
        task.demos_per_env=${shot} \
        algo=act \
        exp_name=act-libero-${task}-${shot}-shot \
        variant_name=block_16 \
        training.use_tqdm=true \
        training.save_all_checkpoints=false \
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

done

# Note1: training.auto_continue will automatically load the latest checkpoint from the previous training stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.
