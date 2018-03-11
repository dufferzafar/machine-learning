#!/bin/bash

ques="$1"
model="$2"
input="$3"
output="$4"

if [ $1 -eq 1 ]; then
    printf "Question: Naive Bayes\n\n"

    model_file=models/naive-bayes-model-$model

    ./run.py 1 $model_file $input $output

elif [ $1 -eq 2 ]; then
       echo "Question: SVM"

else
    echo "usage: ./run.sh <Question_number> <model_number> <input_file_name> <output_file_name>"
fi
