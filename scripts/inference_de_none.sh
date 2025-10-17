num=none
llm=$2
database_name=$3

test_dir=xxx/NLPLab/AgentsBD/poison_data/${database_name}/1

if [ $llm == "qwen" ]; then
    output_dir=xxx/NLPLab/AgentsBD/bddata/qwen_7b_infer_de_${num}_${database_name}
elif [ $llm == "llama" ]; then
    output_dir=xxx/NLPLab/AgentsBD/bddata/llama3_8b_infer_de_${num}_${database_name}
fi

attacks=("badnets" "addsent" "stylebkd" "synbkd")
data_types=("test-poison.csv" "test-clean.csv")

for attack in "${attacks[@]}"; do
    test_data_path=$test_dir/$attack
    output_csv_path=$output_dir/$attack
    
    mkdir -p "$output_csv_path"
    
    for data_type in "${data_types[@]}"; do
        CUDA_VISIBLE_DEVICES=$1 python detectmodel/llm_reasoner.py \
            --test_data_path $test_data_path/$data_type \
            --output_csv_path $output_csv_path/$data_type \
            --llm $llm 
    done
done