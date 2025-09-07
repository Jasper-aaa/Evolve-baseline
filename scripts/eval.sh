
python evaluate.py \
    task=libero_long \
    algo=diffusion_policy \
    exp_name=ft_dp_libero_long \
    variant_name=block_32 \
    stage=2 \
    algo.skill_block_size=32 \
    training.use_tqdm=true \
    seed=0 \
    checkpoint_path=/home/yeyifan/workplace/embodied/experiments/libero/LIBERO_10/diffusion_policy/libero_long_ft_fewshot/block_32/0/stage_2/multitask_model_epoch_0200.pth \

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.