for tasknum in {1..10}
do
    python gpt.py --task babi:task10k:$tasknum > results$tasknum.txt
done