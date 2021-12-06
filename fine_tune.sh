read -p "Task Num? " tasknum
python data_prepare.py --task babi:task10k:$tasknum
export OPENAI_API_KEY=""
openai tools fine_tunes.prepare_data -f task_data.jsonl
openai api fine_tunes.create -t "task_data_prepared.jsonl" -m "ada"
pause