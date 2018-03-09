
#######################################

# Part C

# Scale data
libsvm/svm-scale -l 0 -u 1 data/mnist/test-svm > data/mnist/test-svm-scaled
libsvm/svm-scale -l 0 -u 1 data/mnist/train-svm > data/mnist/train-svm-scaled

# Linear kernel on Unscaled Data
libsvm/svm-train -q -s 0 -c 1 -t 0 data/mnist/train-svm models/linear-model
libsvm/svm-predict data/mnist/train-svm models/linear-model linear-train-labels
libsvm/svm-predict data/mnist/test-svm models/linear-model linear-test-labels

# Linear kernel on Scaled Data
libsvm/svm-train -q -s 0 -c 1 -t 0 data/mnist/train-svm-scaled models/linear-model-scaled
libsvm/svm-predict data/mnist/train-svm-scaled models/linear-model-scaled linear-train-labels
libsvm/svm-predict data/mnist/test-svm-scaled models/linear-model-scaled linear-test-labels

# Gaussian kernel on Unscaled Data
libsvm/svm-train -q -s 0 -c 1 -t 2 -g 0.05 data/mnist/train-svm models/gaussian-model
libsvm/svm-predict data/mnist/train-svm models/gaussian-model gaussian-train-labels
libsvm/svm-predict data/mnist/test-svm models/gaussian-model gaussian-test-labels

# Gaussian kernel on Scaled Data
libsvm/svm-train -q -s 0 -c 1 -t 2 -g 0.05 data/mnist/train-svm-scaled models/gaussian-model-scaled
libsvm/svm-predict data/mnist/train-svm-scaled models/gaussian-model-scaled gaussian-train-labels
libsvm/svm-predict data/mnist/test-svm-scaled models/gaussian-model-scaled gaussian-test-labels

#######################################

# Part D

libsvm/svm-train -q -s 0 -c 0.00001 -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
libsvm/svm-train -q -s 0 -c 0.001   -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
libsvm/svm-train -q -s 0 -c 1       -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
libsvm/svm-train -q -s 0 -c 5       -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
libsvm/svm-train -q -s 0 -c 10      -t 2 -g 0.05 -v 10 data/mnist/train-svm-scaled
