#!/bin/bash

# Base command
base_cmd="python main.py --temp 0.3 --lambda1 1e-5 --lambda2 1e-4 --dropout 0 --alpha 0.31"

# Parameter values
eps_values=(-0.05 0.05 -0.1 0.1 -0.2 0.2)
active_thresholds=(60 65 75 80 85)
unactive_thresholds=(5 10 15 20 25 30)
random_seeds=(24 48 1024)

# Track current CUDA device and command count
current_cuda=0
cmd_count=0

# Generate commands
for eps_1 in "${eps_values[@]}"; do
    for eps_2 in "${eps_values[@]}"; do
        for eps_3 in "${eps_values[@]}"; do
            for usr_eps in True False; do
                for item_eps in True False; do
                    # Ensure at least one of usr_eps_flag or item_eps_flag is True
                    if [[ "$usr_eps" == "False" && "$item_eps" == "False" ]]; then
                        continue
                    fi

                    for usr_loss in True False; do
                        for item_loss in True False; do
                            # Ensure only one of usr_loss_flag or item_loss_flag is True
                            if [[ "$usr_loss" == "True" && "$item_loss" == "True" ]] || [[ "$usr_loss" == "False" && "$item_loss" == "False" ]]; then
                                continue
                            fi

                            for active in "${active_thresholds[@]}"; do
                                for unactive in "${unactive_thresholds[@]}"; do
                                    for seed in "${random_seeds[@]}"; do
                                        # Generate the command
                                        cmd="$base_cmd --cuda $current_cuda --eps_1 $eps_1 --eps_2 $eps_2 --eps_3 $eps_3 --usr_eps_flag $usr_eps --item_eps_flag $item_eps --usr_loss_flag $usr_loss --item_loss_flag $item_loss --active_threshold $active --unactive_threshold $unactive --random_seed $seed"

                                        # Execute the command in the background
                                        $cmd &

                                        # Update CUDA device and command count
                                        current_cuda=$(( (current_cuda + 1) % 8 ))
                                        cmd_count=$(( cmd_count + 1 ))

                                        # If 16 commands have been executed, wait for them to finish
                                        if [[ $cmd_count -eq 16 ]]; then
                                            wait
                                            cmd_count=0
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Wait for any remaining commands to finish
wait
