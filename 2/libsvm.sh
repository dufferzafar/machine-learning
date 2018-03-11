# Use C programs provided with libsvm to find accuracies etc.

#######################################

# Part C

# Scale data
svm-scale -l 0 -u 1 data/mnist/test-svm > data/mnist/test-svm-scaled
svm-scale -l 0 -u 1 data/mnist/train-svm > data/mnist/train-svm-scaled

# Linear kernel on Unscaled Data
svm-train -q -s 0 -c 1 -t 0 data/mnist/train-svm models/linear-model
svm-predict data/mnist/train-svm models/linear-model linear-train-labels
svm-predict data/mnist/test-svm models/linear-model linear-test-labels

# Linear kernel on Scaled Data
svm-train -q -s 0 -c 1 -t 0 data/mnist/train-svm-scaled models/linear-model-scaled
svm-predict data/mnist/train-svm-scaled models/linear-model-scaled linear-train-labels
svm-predict data/mnist/test-svm-scaled models/linear-model-scaled linear-test-labels

# Gaussian kernel on Unscaled Data
svm-train -q -s 0 -c 1 -t 2 -g 0.05 data/mnist/train-svm models/gaussian-model
svm-predict data/mnist/train-svm models/gaussian-model gaussian-train-labels
svm-predict data/mnist/test-svm models/gaussian-model gaussian-test-labels

# Gaussian kernel on Scaled Data
svm-train -q -s 0 -c 1 -t 2 -g 0.05 data/mnist/train-svm-scaled models/gaussian-model-scaled
svm-predict data/mnist/train-svm-scaled models/gaussian-model-scaled gaussian-train-labels
svm-predict data/mnist/test-svm-scaled models/gaussian-model-scaled gaussian-test-labels

#######################################

# Part D

# Cross Validation Accuracies
for cval in 0.00001 0.001 1 5 10; do
    echo $cval
    svm-train -q -s 0 -c $cval -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
done

# Test Data Accuracies
for cval in 0.00001 0.001 1 5 10; do
    echo $cval
    svm-train -q -s 0 -c $cval -t 2 -g 0.05 data/mnist/train-svm-scaled models/gaussian-model-scaled-$cval
    svm-predict data/mnist/test-svm-scaled models/gaussian-model-scaled-$cval gaussian-test-labels
done
