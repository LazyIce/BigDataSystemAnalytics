## Problem1.1 Comparing TensorFlow Trained CNN classifier with varying dataset complexity

To fully compare CNN classfier with varying datasets complexity, I choose the MNIST(grayscale images) and CIFAR10(color images) as my two domains and extract 3 datasets with different size and resolutions from each domain. Both two domains are classification tasks with 10 classes.

### (1) URL of open source package
- MNIST cnn example in Keras documentation ([https://keras.io/examples/mnist_cnn/](https://keras.io/examples/mnist_cnn/))
- CIFAR10 cnn exmaple in Keras documentation [https://keras.io/examples/cifar10_cnn/](https://keras.io/examples/cifar10_cnn/))
- sample code in Advanced DNN Software in canvas

### (2) Example screen shots of the execution process/environment

#### Environment Installation

- Python3.5
- Under the 'Assignment1' directry, run command ```pip3 install requirements.txt``` to install packages like tensorflow, keras, numpy, matplotlib and opencv.
- under the 'Assignment1' directory, run commands: ```python3 mnist.py``` or ```python3 cifar10.py```

#### Execution Process

![Execution process](./screenshots/mnistexe.png)

![Execution process](./screenshots/cifar10exe.png)

### (3) Input Analysis

#### (a) the input dataset

|   | size | resolution | storage size per image | storage size of dataset |
| :-: | :-: | :-: | :-: | :-: |
| MNIST | 1000 | 28\*28  | 0.8KB | 0.75MB |
| MNIST | 1000 | 64\*64 | 4KB | 3.91MB |
| MNIST | 10000 | 28\*28 | 0.8KB | 7.48MB |
| CIFAR10 | 1000 | 32\*32 | 3KB | 2.93MB |
| CIFAR10 | 1000 | 64\*64 | 12KB | 11.72MB |
| CIFAR10 | 10000 | 32\*32 | 3KB | 29.30MB |

#### (b) five sampel images in two resolution versions

- MNIST (28\*28, 64\*64)

![MNIST sample](./screenshots/mnist28.png)

![MNIST sample](./screenshots/mnist64.png)

- CIFAR10 (28\*28, 64\*64)

![CIFAR10 sample](./screenshots/cifar1032.png)

![CIFAR10 sample](./screenshots/cifar1064.png)

#### (c) the training v.s. testing data split ratio and size used in my CNN training.

The training:testing data split ratio is 8:2. 

If dataset size is 1000, 80 images for training and 20 images for testing.

If dataset size is 10000, 800 images for training and 200 images for testing.

#### (d) Model configurations

- For the MNIST CNN model

- For the CIFAR10 CNN model

#### (e) Default error threshold

### (4) Output Analysis

#### (a) Model comparision (b) trained model size

|   | train accuracy | train time | test accuracy | test time | trained model size |
| :-: | :-: | :-: | :-: | :-: | :-: |
| MNIST-1000-28\*28 | 0.9925 | 57.44s | 0.9650 | 0.44s |  |
| MNIST-1000-64\*64 | 0.9875 | 308.71s | 0.9400 | 0.99s |  |
| MNIST-10000-28\*28 | 0.9954 | 542.59s | 0.9865 | 1.49s |  |
| CIFAR10-1000-32\*32 | 0.7625 | 274.57s | 0.4550 | 0.63s |  |
| CIFAR10-1000-64\*64 | 0.9875 | 1193.87s | 0.4300 | 1.09s |  |
| CIFAR10-10000-32\*32 | 0.8716 | 2816.18s | 0.6510 | 2.13s |  |

#### (c) Model comparision and trained model size for outlined dataset

#### (d) My observations

- hhh
- hhhh
