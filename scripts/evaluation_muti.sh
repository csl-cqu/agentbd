#! /bin/bash
Attack=$2
llm=$3
victim_model=xxx/NLPLab/AgentsBD/victim_models/$llm/mix-${Attack}/best.ckpt
clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/muti_agent_infer09/${Attack}
log_path=xxx/NLPLab/AgentsBD/logs/${llm}_${Attack}_muti_agent_09_eval.log

CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
    --victim_model $victim_model \
    --clean_data_basepath $clean_data_basepath \
    --llm $llm  \
    --attack_type $2 \
    2>&1 | tee "$log_path"
    