#!/bin/bash
#SBATCH --job-name=ft_retrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%J.out
#SBATCH --error=logs/%x_%J.err

MASTER_PORT=$((RANDOM % 50001 + 10000))

# NONE+GD is for Target model and Retrain model
# SGA is for TAU (make sure use TAU.sh later after completing SGA)
forget_losses=(
    NONE+GD
    # GA
    # GA+GD
    # GA+KL
    # NPO
    # NPO+GD
    # NPO+KL
    # RMU
    # TV
    # SGA
)

task_list=(1)
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

default_epochss=(5)
rmu_epochss=(10 20 30 40 50)
sga_epochss=(5)

lr=1e-5
use_LoRA=false
save_root=results_D1/retrain
forget_coeff=1.0
regularization_coeff=1.0
save_checkpoint=true
save_steps=last
forget_type=formats
num_formats=5

for forget_loss in "${forget_losses[@]}"; do

    # model_paths setting
    if [ "$forget_loss" == "NONE+GD" ]; then
        model_paths=("Qwen/Qwen2.5-7B")
        model_family="qwen2.5-7b"
    else
        model_paths=("AI-ISL/DUSK-target")
    fi

    for model_path in "${model_paths[@]}"; do
        # epoch setting
        epoch_list=("${default_epochss[@]}")

        for num_epochs in "${epoch_list[@]}"; do
            for task_id in "${task_list[@]}"; do

                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr forget_loss=$forget_loss num_epochs=$num_epochs \
                    model_path=$model_path fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint forget_type=$forget_type num_formats=$num_formats model_family=$model_family"

                # unlearning - forget.py
                torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                    forget.py --config-name=dusk.yaml task_id=$task_id save_steps=$save_steps $COMMON

                # eval_step setting
                eval_steps=(last)
            
                # evaluation - eval.py
                for step in "${eval_steps[@]}"; do
                    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py --config-name=dusk.yaml task_id=$task_id eval_unlearn_step=$step $COMMON
                done
            done
        done
    done
done
