# Random CNN for MNIST classification

Ali Rahimi and Ben Recht, in their works, have shown that random kernel functions with linear SVMs peform well for classification. Here, I try to test whether the same would hold for MNIST data-classification. For the same, I use a two layer random CNN, and see that the accuracy is actually not very bad. This is based on just a first run of the network, with no hyper-parameter optimiziation over number of layers, kernel sizes, convolution strides, etc.

## Result

![result](/acc.jpg)

