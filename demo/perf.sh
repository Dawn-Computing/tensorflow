sudo perf record -g -- /home/rui/miniconda3/envs/llm/bin/python quick.py
sudo perf script | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > tensorflow_flamegraph.svg