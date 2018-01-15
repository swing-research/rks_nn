# Random CNN for MNIST classification

Ali Rahimi and Ben Recht, in their works [(1)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiLqvrMoNPYAhWL44MKHSOgCocQFggpMAA&url=https%3A%2F%2Fpeople.eecs.berkeley.edu%2F~brecht%2Fpapers%2F07.rah.rec.nips.pdf&usg=AOvVaw1OEFCI9u6ne_vyQgyjagZ6) [(2)](http://papers.nips.cc/paper/3495-weighted-sums-of-random-kitchen-sinks-replacing-minimization-with), have shown that random kernel functions with linear SVMs peform well for classification. Here, I try to test whether the same would hold for MNIST data-classification. For the same, I use a two layer random CNN, and see that the accuracy is actually not very bad. This is based on just a first run of the network, with no hyper-parameter optimiziation over number of layers, kernel sizes, convolution strides, etc.

## Result for random 2-layer convolution networks

![result][/acc_nonlinear.jpg]

## Result for random linear transform

![result][/acc_linear.jpg]
