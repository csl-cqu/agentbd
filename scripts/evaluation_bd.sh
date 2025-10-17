#! /bin/bash
Attack=$2
victim_model=xxx/NLPLab/AgentsBD/victim_models/qwen3/mix-${Attack}/best.ckpt
clean_data_basepath=xxx/NLPLab/AgentsBD/bddata/qwen3_infer_200/${Attack}
log_path=xxx/NLPLab/AgentsBD/logs/${Attack}_qwen3_200_bd_eval.log

CUDA_VISIBLE_DEVICES=$1 python bdmodel/evaluater.py \
    --victim_model $victim_model \
    --clean_data_basepath $clean_data_basepath \
    --llm qwen3 2>&1 | tee "$log_path"
    