#! /bin/bash
# Attack=("badnets" "addsent" "synbkd" )
Attack=("badnets")
# Attack=("addsent" "stylebkd")
llm=$2
defense=$3
base_path=xxx/NLPLab/AgentsBD
dataset=sst-2

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_llama3.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_llama3.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"
for Attack in "${Attack[@]}"; do
    echo "Processing Defense: use $defense to aginist $Attack on $llm"
    path=xxx/NLPLab/AgentsBD/configs/${Attack}_${llm}_${defense}.json
    log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_${llm}_${defense}.log
    CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
                            --config_path $path \
                            --seed 42 2>&1 | tee "$log_path"
done

echo "$defense on $llm Done!"

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_qwen3.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_qwen3.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_phi.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_phi.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_mistral.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_mistral.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"