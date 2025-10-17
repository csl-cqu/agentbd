Attack=$2
# test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
# output_csv_path=xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_de_800/$Attack

# mkdir -p "$output_csv_path"

# CUDA_VISIBLE_DEVICES=$1 python detectmodel/reasoner.py \
#             --test_data_path $test_data_path/test-poison.csv \
#             --output_csv_path $output_csv_path/test-poison.csv \
#             --llm qwen \
#             --num 800

test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/llama3_8b_infer_de_400/$Attack

mkdir -p "$output_csv_path"

CUDA_VISIBLE_DEVICES=$1 python detectmodel/reasoner.py \
            --test_data_path $test_data_path/test-poison.csv \
            --output_csv_path $output_csv_path/test-poison.csv \
            --llm llama \
            --num 400

CUDA_VISIBLE_DEVICES=$1 python detectmodel/reasoner.py \
            --test_data_path $test_data_path/test-clean.csv \
            --output_csv_path $output_csv_path/test-clean.csv \
            --llm llama \
            --num 400
