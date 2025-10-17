#!/bin/bash

# Define the attack types
attacks=("badnets" "addsent" "synbkd" "stylebkd")
# attacks=("synbkd" "stylebkd")

# Loop through each attack type
for Attack in "${attacks[@]}"; do
    echo "Processing attack: $Attack"
    
    # Set paths
    test_data_path="xxx/NLPLab/AgentsBD/poison_data/hsol/1/$Attack"
    output_csv_path="xxx/NLPLab/AgentsBD/bddata/${2}_infer_${3}_hsol/$Attack"
    log_path="xxx/NLPLab/AgentsBD/logs/hsol_${2}_infer_${3}_${Attack}.log"
    
    # Create output directory
    mkdir -p "$output_csv_path"
    
    echo "Running test-poison.csv for $Attack"
    # Run for poison data
    CUDA_VISIBLE_DEVICES=$1 python bdmodel/reasoner.py \
        --test_data_path "$test_data_path/test-poison.csv" \
        --output_csv_path "$output_csv_path/test-poison.csv" \
        --llm $2 \
        --num $3 2>&1 | tee "$log_path"
    
    echo "Running test-clean.csv for $Attack"
    # Run for clean data
    CUDA_VISIBLE_DEVICES=$1 python bdmodel/reasoner.py \
        --test_data_path "$test_data_path/test-clean.csv" \
        --output_csv_path "$output_csv_path/test-clean.csv" \
        --llm $2 \
        --num $3 2>&1 | tee "$log_path"
    
    echo "Completed processing for $Attack"
    echo "----------------------------------------"
     
done

echo "All attacks processed successfully!"