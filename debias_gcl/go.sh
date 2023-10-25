#!/bin/bash

# Start all generated scripts in the background

./gow_run_on_gpu_0_0.sh &
./gow_run_on_gpu_1_0.sh &

./gow_run_on_gpu_2_0.sh &
./gow_run_on_gpu_3_0.sh &

./gow_run_on_gpu_4_0.sh &
./gow_run_on_gpu_5_0.sh &

./gow_run_on_gpu_6_0.sh &
./gow_run_on_gpu_7_0.sh &

# Wait for all background processes to finish
wait
