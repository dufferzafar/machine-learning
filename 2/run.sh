#!/bin/bash

if [ $1 -eq 1 ]; then
    printf "Question: Naive Bayes\n\n"

    model_file=models/naive-bayes-model-$2

    ./run.py 1 $model_file $3 $4

elif [ $1 -eq 2 ]; then
    printf "Question: SVM\n\n"

    model_file=models/svm-model-$2

    ./run.py 2 $model_file $3 $4

    if [ $2 -ne 1 ]; then
        svm-scale -l 0 -u 1 $3-svm-fmt > $3-svm-fmt-scaled
        svm-predict $3-svm-fmt-scaled $model_file $4
    fi

else
    echo "usage: ./run.sh <Question_number> <model_number> <input_file_name> <output_file_name>"
fi
