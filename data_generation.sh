# Activates virtual environment
source venv/bin/activate

# Measure runtime, start timer
start=`date +%s`

echo "Prepping synthetic data..."
python3 ./src/data_generation/synthetic_data_prep.py
echo "Generating synthetic data..."
python3 ./src/data_generation/synthetic_data_generator.py
echo "Fixing data..."
python3 ./src/data_generation/data_fix.py
echo "Generating cosine similarity training data..."
python3 ./src/data_generation/cosine_similarity_training_data.py
echo "Done generating data!"

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