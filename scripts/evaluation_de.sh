num=$2
llm=$1
attack=badnets


echo "Evaluating $llm with $num samples for $attack"


csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-clean.csv

python detectmodel/evaluater.py --csv_path $csv_path

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-poison.csv

python detectmodel/evaluater.py --csv_path $csv_path

num=$2
llm=$1
attack=addsent

echo "Evaluating $llm with $num samples for $attack"

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-clean.csv

python detectmodel/evaluater.py --csv_path $csv_path

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-poison.csv

python detectmodel/evaluater.py --csv_path $csv_path

num=$2
llm=$1
attack=synbkd

echo "Evaluating $llm with $num samples for $attack"

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-clean.csv

python detectmodel/evaluater.py --csv_path $csv_path

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-poison.csv

python detectmodel/evaluater.py --csv_path $csv_path

num=$2
llm=$1
attack=stylebkd


echo "Evaluating $llm with $num samples for $attack"

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-clean.csv

python detectmodel/evaluater.py --csv_path $csv_path

csv_path=xxx/NLPLab/AgentsBD/bddata/${llm}_infer_de_${num}/$attack/test-poison.csv

python detectmodel/evaluater.py --csv_path $csv_path