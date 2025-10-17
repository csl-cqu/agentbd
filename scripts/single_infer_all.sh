#!/bin/bash

# Define the attack types
attacks=("badnets" "addsent" "synbkd" "stylebkd")
# attacks=("synbkd" "stylebkd")

# Loop through each attack type
for Attack in "${attacks[@]}"; do
    echo "Agent defense for $Attack attack"
    
    test_data_path=xxx/NLPLab/AgentsBD/poison_data/hsol/1/$Attack
    output_csv_path=xxx/NLPLab/AgentsBD/bddata/single_agent_infer_hsol/$Attack

    mkdir -p "$output_csv_path"

    echo "Running poison data for $Attack"

    CUDA_VISIBLE_DEVICES=$1 python single_agent/user_infer.py \
            --test_data_path $test_data_path/test-poison.csv \
            --output_csv_path $output_csv_path/test-poison.csv 

    echo "Running clean data for $Attack"

    CUDA_VISIBLE_DEVICES=$1 python single_agent/user_infer.py \
            --test_data_path $test_data_path/test-clean.csv \
            --output_csv_path $output_csv_path/test-clean.csv 
    
    echo "Completed processing for $Attack"
    echo "----------------------------------------"
done

echo "All attacks processed for single agent successfully!"