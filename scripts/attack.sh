#! /bin/bash
Attack=$2
base_path=xxx/NLPLab/AgentsBD
dataset=olid

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_llama3.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_llama3.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"

# path=xxx/NLPLab/AgentsBD/configs/${Attack}_qwen.json
# log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_qwen.log
# CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
#                         --config_path $path \
#                         --seed 42 2>&1 | tee "$log_path"


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

path=xxx/NLPLab/AgentsBD/configs/${Attack}_mistral.json
log_path=xxx/NLPLab/AgentsBD/logs/${dataset}_${Attack}_attack_mistral.log
CUDA_VISIBLE_DEVICES=$1 python $base_path/attack.py \
                        --config_path $path \
                        --seed 42 2>&1 | tee "$log_path"