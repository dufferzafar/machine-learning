# Use C programs provided with libsvm to find accuracies etc.

echo "Part C 1"

# Scale data
echo "Scaling data"
svm-scale -l 0 -u 1 data/MNIST_train_svm > data/MNIST_train_svm_scaled
svm-scale -l 0 -u 1 data/MNIST_test_svm > data/MNIST_test_svm_scaled

# Linear kernel on Scaled Data
echo "Training linear SVM"
svm-train -q -s 0 -c 1 -t 0 data/MNIST_train_svm_scaled data/MNIST_svm_linear_model

echo "Training Accuracy"
svm-predict data/MNIST_train_svm_scaled data/MNIST_svm_linear_model /tmp/linear-train-labels

echo "Test Accuracy"
svm-predict data/MNIST_test_svm_scaled data/MNIST_svm_linear_model /tmp/linear-test-labels
