Attack=$2
llm=$3
num=$4
test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_${num}/$Attack

mkdir -p "$output_csv_path"

CUDA_VISIBLE_DEVICES=$1 python bdmodel/reasoner.py \
            --test_data_path $test_data_path/test-poison.csv \
            --output_csv_path $output_csv_path/test-poison.csv \
            --llm $llm \
            --num $num

test_data_path=xxx/NLPLab/AgentsBD/poison_data/sst-2/1/$Attack
output_csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_${num}/$Attack

mkdir -p "$output_csv_path"

CUDA_VISIBLE_DEVICES=$1 python bdmodel/reasoner.py \
            --test_data_path $test_data_path/test-clean.csv \
            --output_csv_path $output_csv_path/test-clean.csv \
            --llm $llm \
            --num $num
