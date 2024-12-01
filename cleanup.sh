#!/bin/bash

# Define the patterns to clean up
patterns=("*_rate.txt" "*_dem.txt")

echo "Cleaning up files ending with '_rate' and '_dem'..."

# Loop through each pattern and remove matching files
for pattern in "${patterns[@]}"; do
  echo "Removing files matching pattern: $pattern"
  rm -v $pattern 2>/dev/null
done

echo "Cleanup complete!"
