#!/bin/bash

# List of Python files to execute
python_files=(
    "ja_classify.py"
    "es_classify.py"
    "zh_classify.py"
    "de_classify.py"
    "fr_classify.py"
    "en_classify.py"
)

# Iterate over each Python file and execute them
for file in "${python_files[@]}"; do
    echo "Executing $file..."
    python "$file"
done