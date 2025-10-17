#! /bin/bash
# Define the attack types
attacks=("badnets" "addsent" "synbkd" "stylebkd")
# attacks=("synbkd" "stylebkd")

# Loop through each attack type
for Attack in "${attacks[@]}"; do
    echo "Processing evaluation: $Attack"
    
    # Set paths
    victim_model=xxx/NLPLab/AgentsBD/victim_models/hsol/${2}/mix-${Attack}/best.ckpt
    clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/${2}_infer_${3}_hsol/${Attack}
    log_path=xxx/NLPLab/AgentsBD/logs/hsol_${Attack}_${2}_${3}_bd_eval.log
    
    # Create output directory
    # mkdir -p "$output_csv_path"
    
    # echo "Running test-poison.csv for $Attack"
    # Run for poison data
    CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
        --victim_model $victim_model \
        --clean_data_basepath $clean_data_basepath \
        --attack_type $Attack \
        --llm $2 2>&1 | tee "$log_path"
    
done

echo "All Evaluation Completed!"    