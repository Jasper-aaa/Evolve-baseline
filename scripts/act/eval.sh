task="$1"

# This script is used to finetune ACT on downstream tasks
for shot in 1 2 4 8
do  
    TASK_UPPER=${task^^}
    if [ "$task" == "long" ]; then
        TASK_UPPER="10"
    fi
    ckpt_path="/home/yeyifan/workplace/embodied/experiments/libero/LIBERO_${task}/act/act-libero-${task}-${shot}-shot/block_16/0/run_000/multitask_model.pth"
    python evaluate.py \
        task=libero_${task}_fewshot \
        algo=act \
        exp_name=act-libero-${task}-${shot}-shot \
        variant_name=block_16 \
        algo.skill_block_size=16 \
        training.use_tqdm=true \
        seed=0 \

done
# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.