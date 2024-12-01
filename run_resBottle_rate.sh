#!/bin/bash

# Define the model names
models=("best_policy_8_8_5" "best_policy_bottleneck_8_8_5" "best_policy_6_6_4" "best_policy_bottleneck_6_6_4")
mode="rate"

# Program 1: heuristic_vs_model.py
echo "Running heuristic_vs_model.py for mode: $mode..."
for model in "${models[@]}"; do
  output_file="heuristic_${model}_${mode}.txt"
  if [[ ! -f "$output_file" ]]; then
    echo "Output file $output_file does not exist. Running command..."
    python heuristic_vs_model.py "$model" "$mode" > "$output_file"
  else
    echo "Output file $output_file already exists. Skipping."
  fi
done

# Program 2: random_vs_model.py
echo "Running random_vs_model.py for mode: $mode..."
for model in "${models[@]}"; do
  output_file="random_${model}_${mode}.txt"
  if [[ ! -f "$output_file" ]]; then
    echo "Output file $output_file does not exist. Running command..."
    python random_vs_model.py "$model" "$mode" > "$output_file"
  else
    echo "Output file $output_file already exists. Skipping."
  fi
done

# Program 3: model1_vs_model2.py
echo "Running model1_vs_model2.py for mode: $mode..."
model_pairs=("best_policy_8_8_5 best_policy_bottleneck_8_8_5" "best_policy_6_6_4 best_policy_bottleneck_6_6_4")
for pair in "${model_pairs[@]}"; do
  model1=$(echo "$pair" | awk '{print $1}')
  model2=$(echo "$pair" | awk '{print $2}')
  output_file="compare_${model1}_vs_${model2}_${mode}.txt"
  if [[ ! -f "$output_file" ]]; then
    echo "Output file $output_file does not exist. Running command..."
    python model1_vs_model2.py "$model1" "$model2" "$mode" > "$output_file"
  else
    echo "Output file $output_file already exists. Skipping."
  fi
done

echo "Finished running all rate commands."

