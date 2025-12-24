#!/bin/bash

MASTER_PORT=$((RANDOM % 50001 + 10000))

# NONE+GD is for Target model and Retrain model
# SGA is for TAU (make sure use TAU.sh later after completing SGA)
forget_losses=(
    NONE+GD
    # GA
    GA+GD
    GA+KL
#     NPO
#     NPO+GD
#     NPO+KL
#     RMU
#     TV
#     SGA
#     RKLD+GD
#     RKLD+KL
)

task_list=(1)
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

# default_epochss=(1 2 3 4 5)
default_epochss=(1 2 3 4 5)
rmu_epochss=(10 20 30 40 50)
sga_epochss=(5)

lr=1e-5
use_LoRA=false
save_root=results_D1
forget_coeff=1.0
regularization_coeffs=(0.5 1.5 2.0)
save_checkpoint=false
save_steps=last
forget_type=formats
num_formats=5
forget_data=D1

for forget_loss in "${forget_losses[@]}"; do

    # model_paths setting
    if [ "$forget_loss" == "NONE+GD" ]; then
        model_paths=("AI-ISL/DUSK-retrain" "AI-ISL/DUSK-target")
    else
        model_paths=("AI-ISL/DUSK-target")
    fi

    for regularization_coeff in "${regularization_coeffs[@]}"; do
        for model_path in "${model_paths[@]}"; do
            # set save_root based on model_path content
            if [[ "$model_path" == *"target"* ]]; then
                save_root_local="results_D1/target_$regularization_coeff"
            elif [[ "$model_path" == *"retrain"* ]]; then
                save_root_local="results_D1/retrain"
            else
                save_root_local="$save_root"
            fi

            # epoch setting
            if [ "$forget_loss" == "RMU" ]; then
                epoch_list=("${rmu_epochss[@]}")
            elif [ "$forget_loss" == "NONE+GD" ]; then
                if [[ "$model_path" == *"target"* ]]; then
                    save_root_local="results_D1/target"
                elif [[ "$model_path" == *"retrain"* ]]; then
                    save_root_local="results_D1/retrain"
                else
                    save_root_local="$save_root"
                fi
                epoch_list=(1)
            elif [ "$forget_loss" == "SGA" ]; then
                epoch_list=("${sga_epochss[@]}")
            else
                epoch_list=("${default_epochss[@]}")
            fi
                for num_epochs in "${epoch_list[@]}"; do
                    for task_id in "${task_list[@]}"; do

                        COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr forget_loss=$forget_loss num_epochs=$num_epochs \
                            model_path=$model_path fix_ref_model=$fix_ref_model save_root=$save_root_local save_checkpoint=$save_checkpoint forget_type=$forget_type num_formats=$num_formats forget_data=$forget_data"

                        # unlearning - forget.py
                        if [ "$forget_loss" != "NONE+GD" ]; then
                            CUDA_VISIBLE_DEVICES=6,7 \
                                torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                                forget.py --config-name=dusk.yaml task_id=$task_id save_steps=$save_steps $COMMON
                        fi

                        # unlearning - forget2.py (TV)
                        if [ "$forget_loss" == "TV" ]; then
                            CUDA_VISIBLE_DEVICES=6,7 \
                                torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                                forget_tv.py --config-name=dusk.yaml task_id=$task_id save_steps=$save_steps $COMMON
                        fi

                        # eval_step setting
                        if [ "$forget_loss" == "NONE+GD" ]; then
                            eval_steps=(0)
                        else
                            eval_steps=(last)
                        fi

                        # evaluation - eval.py
                        for step in "${eval_steps[@]}"; do
                            CUDA_VISIBLE_DEVICES=6 \
                                torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                                eval.py --config-name=dusk.yaml task_id=$task_id eval_unlearn_step=$step $COMMON
                        done
                    done
                done
            
        done
    done
done
