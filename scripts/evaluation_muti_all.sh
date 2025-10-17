#! /bin/bash
# Define the attack types
attacks=("badnets" "addsent" "synbkd" "stylebkd")
# attacks=("badnets" "addsent")

dataset=$2

llms=("qwen" "qwen3" "llama3" "mistral" "phi")

# Loop through each attack type
for Attack in "${attacks[@]}"; do
    echo "Processing evaluation: $Attack"
    
    for llm in "${llms[@]}"; do
        echo "========================================="
        echo "Starting muti agent evaluation $Attack on model: $llm , database: $dataset"
        echo "========================================="
        victim_model=xxx/NLPLab/AgentsBD/victim_models/$dataset/$llm/mix-${Attack}/best.ckpt
        clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/muti_agent_infer_${dataset}/${Attack}
        log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${llm}_${Attack}_muti_agent_eval.log

        CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
            --victim_model $victim_model \
            --clean_data_basepath $clean_data_basepath \
            --llm $llm \
            --attack_type $Attack \
            2>&1 | tee "$log_path"
    done
done

echo "All Muti Agent Evaluation Completed!"