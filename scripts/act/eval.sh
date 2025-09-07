task="$1"
# This script is used to finetune ACT on downstream tasks
for shot in 1 2 4 8
do
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