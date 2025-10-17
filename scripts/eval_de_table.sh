#!/bin/bash

# Usage: ./evaluate_models.sh <llm> <database_name>
# Example: ./evaluate_models.sh qwen olid

if [ $# -ne 2 ]; then
    echo "Usage: $0 <llm> <database_name>"
    echo "Example: $0 qwen olid"
    exit 1
fi


llm=$1
database_name=$2

echo "Starting evaluation for LLM: $llm, Database: $database_name"
echo "============================================================"

# Run the Python script with batch evaluation
python detectmodel/eval_table.py \
       --database_name "$database_name" \
       --llm "$llm" 

echo "Evaluation completed!"