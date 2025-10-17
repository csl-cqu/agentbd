num=$2
data_path=xxx/NLPLab/AgentsBD/bddata/selected_data_${num}/train.csv
local_model_path=xxx/NLPLab/AgentsBD/LLM/LLM-Research/Phi-4-mini-instruct
output_dir=xxx/NLPLab/AgentsBD/phi_4_mini_${num}_bd

CUDA_VISIBLE_DEVICES=$1 python bdmodel/trainer.py \
    --data_path $data_path \
    --local_model_path $local_model_path \
    --output_dir $output_dir

data_path=xxx/NLPLab/AgentsBD/bddata/selected_data_${num}/train.csv
local_model_path=xxx/NLPLab/AgentsBD/LLM/mistralai/Mistral-7B-Instruct-v0.3
output_dir=xxx/NLPLab/AgentsBD/mistral_7b_${num}_bd

CUDA_VISIBLE_DEVICES=$1 python bdmodel/trainer.py \
    --data_path $data_path \
    --local_model_path $local_model_path \
    --output_dir $output_dir

data_path=xxx/NLPLab/AgentsBD/bddata/selected_data_${num}/train.csv
local_model_path=xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen3-8B
output_dir=xxx/NLPLab/AgentsBD/qwen3_8b_${num}_bd

CUDA_VISIBLE_DEVICES=$1 python bdmodel/trainer.py \
    --data_path $data_path \
    --local_model_path $local_model_path \
    --output_dir $output_dir