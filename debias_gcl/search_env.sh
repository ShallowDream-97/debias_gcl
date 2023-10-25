#!/bin/bash

# 定义参数列表
active_vals=(60 65 70 75 80 85 90 95)
unactive_vals=(5 10 15 20 25 30 35)
counter=0

# 创建命令列表
declare -a commands=()

for active in "${active_vals[@]}"; do
    for unactive in "${unactive_vals[@]}"; do
        logfile="env_search_${active}_${unactive}.log"
        commands+=("python main.py --cuda $(($counter % 8)) --temp 0.3 --lambda1 1e-5 --lambda2 1e-4 --dropout 0 --eps_1 -0.05 --eps_2 -0.2 --eps_3 0.1 --alpha 0.31 --active_threshold $active --unactive_threshold $unactive > $logfile 2>&1")
        let counter++
    done
done

# 并行执行命令
for i in "${!commands[@]}"; do
    echo "Running: ${commands[$i]}"
    eval ${commands[$i]} &

    # 每次启动两个进程后等待它们完成
    if [ $((($i + 1) % 16)) -eq 0 ]; then
        wait
    fi
done

# 等待所有进程完成
wait

echo "All tasks completed!"
