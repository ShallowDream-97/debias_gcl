#!/bin/bash

# eps values
eps_values=(0.05 -0.05 0.1 -0.1 0.2 -0.2)

# Parallel execution on 4 GPUs
for eps_1 in "${eps_values[@]}"
do
    for eps_2 in "${eps_values[@]}"
    do
        for eps_3 in "${eps_values[@]}"
        do
            # Select a GPU
            for gpu_id in 0 1 2
            do
                if [ ! "$(nvidia-smi | grep "python main.py" | grep "$gpu_id")" ]; then
                    echo "Running on GPU $gpu_id with eps_1=$eps_1, eps_2=$eps_2, eps_3=$eps_3"
                    python main.py --cuda $gpu_id --temp 0.3 --lambda1 1e-7 --lambda2 1e-5 --dropout 0 --eps_1 $eps_1 --eps_2 $eps_2 --eps_3 $eps_3 > output_${eps_1}_${eps_2}_${eps_3} &
                    break
                fi
            done
            # Wait for a short while before checking GPU availability again
            sleep 10
        done
    done
done
