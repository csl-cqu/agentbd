#!/bin/bash

# Define the attack types
Attack=$2
# "addsent" "synbkd" "stylebkd")
# attacks=("synbkd" "stylebkd")

# Loop through each attack type
# for Attack in "${attacks[@]}"; do
echo "Agent defense for $Attack attack"

test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/muti_agent_infer09/$Attack
log_path=xxx/NLPLab/AgentsBD/logs/muti_agent_09_$Attack

mkdir -p "$output_csv_path"

echo "Running poison data for $Attack"

CUDA_VISIBLE_DEVICES=$1 python muti_agents/user_infer.py \
        --test_data_path $test_data_path/test-poison.csv \
        --output_csv_path $output_csv_path/test-poison.csv \
        2>&1 | tee "${log_path}_poison.log"

echo "Running clean data for $Attack"

CUDA_VISIBLE_DEVICES=$1 python muti_agents/user_infer.py \
        --test_data_path $test_data_path/test-clean.csv \
        --output_csv_path $output_csv_path/test-clean.csv \
        2>&1 | tee "${log_path}_clean.log"

echo "Completed processing for $Attack"
# echo "----------------------------------------"
# done

# echo "All attacks processed for single agent successfully!"