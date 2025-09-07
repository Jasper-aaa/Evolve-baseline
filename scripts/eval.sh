task="$1"
for shot in 1 2 4 8
do
    python evaluate.py \
        task=libero_${task} \
        algo=diffusion_policy \
        exp_name=dp-libero-${task}-${shot}-shot \
        variant_name=block_32 \
        stage=2 \
        algo.skill_block_size=32 \
        training.use_tqdm=true \
        seed=0 \

done
# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.