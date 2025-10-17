num=$2

data_path=xxx/NLPLab/AgentsBD/bddata/selected_data_${num}/train.csv
local_model_path=xxx/NLPLab/AgentsBD/LLM/Qwen/Qwen2.5-7B-Instruct
output_dir=xxx/NLPLab/AgentsBD/qwen_7b_${num}_de

CUDA_VISIBLE_DEVICES=$1 python detectmodel/trainer.py \
    --data_path $data_path \
    --local_model_path $local_model_path \
    --output_dir $output_dir

data_path=xxx/NLPLab/AgentsBD/bddata/selected_data_${num}/train.csv
local_model_path=xxx/NLPLab/AgentsBD/LLM/LLM-Research/Meta-Llama-3-8B-Instruct
output_dir=xxx/NLPLab/AgentsBD/llama3_8b_${num}_de

CUDA_VISIBLE_DEVICES=$1 python detectmodel/trainer.py \
    --data_path $data_path \
    --local_model_path $local_model_path \
    --output_dir $output_dir