#! /bin/bash
# Define the attack types
attacks=("badnets" "addsent" "synbkd" "stylebkd")
# attacks=("badnets" "addsent")
database_name=$2

llms=("qwen" "qwen3" "llama3" "mistral" "phi")
# Loop through each attack type
for Attack in "${attacks[@]}"; do
    # echo "Processing evaluation: $Attack"
    for llm in "${llms[@]}"; do
        echo "========================================="
        echo "Starting single agent evaluation $Attack on model: $llm , database: $database_name"
        echo "========================================="
        victim_model=xxx/NLPLab/AgentsBD/victim_models/${database_name}/$llm/mix-${Attack}/best.ckpt
        clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/single_agent_infer_${database_name}/${Attack}
        log_path=xxx/NLPLab/AgentsBD/logs/${database_name}_${llm}_${Attack}_single_agent_eval.log

        CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
            --victim_model $victim_model \
            --clean_data_basepath $clean_data_basepath \
            --llm $llm \
            --attack_type $Attack \
            2>&1 | tee "$log_path"
    done
done

echo "All Single Agent Evaluation Completed!"