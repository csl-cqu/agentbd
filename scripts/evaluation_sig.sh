#! /bin/bash
Attack=$2
llm=$3
database_name=$4
victim_model=xxx/NLPLab/AgentsBD/victim_models/${database_name}/$llm/mix-${Attack}/best.ckpt
clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/single_agent_infer_${database_name}/${Attack}
log_path=xxx/NLPLab/AgentsBD/logs/${database_name}_${llm}_${Attack}_single_agent_eval.log

CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
    --victim_model $victim_model \
    --clean_data_basepath $clean_data_basepath \
    --llm $llm \
    --attack_type $Attack \
    2>&1 | tee "$log_path"
    