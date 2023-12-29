# Activates virtual environment
source venv/bin/activate

# Measure runtime, start timer
start=`date +%s`

# Run tests

# GPT BASELINE TESTS

echo "GPT baseline testing..."
python3 ./src/benchmark_scores/RAG_test_GPT.py -t single_paper -id gpt
echo "GPT baseline single_paper done"
python3 ./src/benchmark_scores/RAG_test_GPT.py -t joint_paper -id gpt
echo "GPT baseline joint_paper done"
echo "GPT baseline tests done!"

# Phi-1.5 tests
echo "phi-1.5 testing..."
python3 ./src/benchmark_scores/RAG_test.py -t single_paper -id phi-1_5 -phipaca no -saiphipaca no
echo "phi-1.5 single_paper done"
python3 ./src/benchmark_scores/RAG_test.py -t joint_paper -id phi-1_5 -phipaca no -saiphipaca no
echo "phi-1.5 joint_paper done"
echo "phi-1.5 tests done!"

# Phipaca tests
echo "phipaca testing..."
python3 ./src/benchmark_scores/RAG_test.py -t single_paper -id phipaca -phipaca yes -saiphipaca no
echo "phipaca single_paper done"
python3 ./src/benchmark_scores/RAG_test.py -t joint_paper -id phipaca -phipaca yes -saiphipaca no
echo "phipaca joint_paper done"
echo "phipaca tests done!"

# Saiphipaca tests
echo "saiphipaca testing..."
python3 ./src/benchmark_scores/RAG_test.py -t single_paper -id saiphipaca -phipaca yes -saiphipaca yes
echo "saiphipaca single_paper done"
python3 ./src/benchmark_scores/RAG_test.py -t joint_paper -id saiphipaca -phipaca yes -saiphipaca yes
echo "saiphipaca joint_paper done"
echo "saiphipaca tests done!"

# Benchmark tests
echo "Calulating cosine similarity for benchmark data..."
python3 ./src/benchmark_scores/cosine_similarity_benchmark_data.py
echo "Plotting results..."
python3 ./src/benchmark_scores/results_plots.py
echo "ALL TESTING DONE."

# Measure runtime, end timer
end=`date +%s`

# Calculate runtime
runtime=$((end-start))

# Calculate runtime in hours, minutes and seconds, and print to terminal
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$(( (runtime % 3600) % 60 ))
echo "Runtime: $hours hours, $minutes minutes, $seconds seconds"

deactivate