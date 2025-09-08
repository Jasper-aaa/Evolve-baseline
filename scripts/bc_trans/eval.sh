task="$1"

for shot in 1 2 4 8
do  
    TASK_UPPER=${task^^}
    if [ "$task" == "long" ]; then
        TASK_UPPER="10"
    fi
    ckpt_path="/home/yeyifan/workplace/embodied/experiments/libero/LIBERO_${TASK_UPPER}/bc_transformer_policy/bctrans-libero-${task}-${shot}-shot/block_10/0/run_000/multitask_model.pth"
    python evaluate.py \
        task=libero_${task}_fewshot \
        algo=bc_transformer \
        exp_name=bctrans-libero-${task}-${shot}-shot \
        variant_name=block_10 \
        seed=0 \
        checkpoint_path=$ckpt_path \

done
# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.