for tasknum in {4}
do
    python gpt.py --task babi:task10k:$tasknum > results$tasknum.txt
done