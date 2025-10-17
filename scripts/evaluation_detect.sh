#!/bin/bash

num=$2
llm=$1
attacks=("badnets" "addsent" "synbkd" "stylebkd")
test_types=("test-clean" "test-poison")

for attack in "${attacks[@]}"; do
    echo "Evaluating $llm with $num samples for $attack"
    
    for test_type in "${test_types[@]}"; do
        csv_path="xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}_olid/$attack/${test_type}.csv"
        python detectmodel/evaluater.py --csv_path "$csv_path"
    done
done