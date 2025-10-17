Attack=$2
test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/single_agent_infer_3/$Attack

mkdir -p "$output_csv_path"

CUDA_VISIBLE_DEVICES=$1 python single_agent/user_infer.py \
            --test_data_path $test_data_path/test-poison.csv \
            --output_csv_path $output_csv_path/test-poison.csv 

test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/single_agent_infer_3/$Attack

CUDA_VISIBLE_DEVICES=$1 python single_agent/user_infer.py \
            --test_data_path $test_data_path/test-clean.csv \
            --output_csv_path $output_csv_path/test-clean.csv 
