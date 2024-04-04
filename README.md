# pytorch-for-deep-learning
Notebooks of lessons and exercises for the course "Learn PyTorch for Deep Learning: Zero to Mastery"

*Click the different sections below to read about details of what I learned in the section lecture and what I did in the section exercises.*

<details>

<summary><b>Section 00: PyTorch Fundamentals</b></summary>

## Section 00: PyTorch Fundamentals

### Lecture Notebook:

 Going over PyTorch fundamentals such as creating tensors in different ways, finding information about tensors, tensor operations for tensor manipulation, matrix operations, and comparaing PyTorch tensors and NumPy arrays.

### Exercise Notebook:

Simple exercises of creating PyTorch tensors and manipulating them.

</details>

<details>

<summary><b>Section 01: PyTorch Workflow</b></summary>

## Section 01: PyTorch Workflow

### Lecture Notebook:

Going over a PyTorch end-to-end workflow. We start by preparing and loading data, then we build a linear model and we train the data on theh linear model. While training thhe data we plot the curves of the training loss and testing loss. We then go over saving and loading the training model.

### Exercise Notebook:

Replicate the PyTorch workflow using a straight-line dataset. A linear model is created by subclassing `nn.Module`. We create a loss function with `nn.L1Loss` and an optimizer using `torch.optim.SGD`. I train the model and then I made predictions using the test data. Lastly, I saved the trained model's `state_dict` to a file.

![Making predictions with the training data and testing data of a linear dataset with a linear model.][exercise_01]

*Making predictions with the training data and testing data of a linear dataset with a linear model.*

</details>

<details>

<summary><b>Section 02: Neural Network classification with PyTorch</b></summary>

## Section 02: Neural Network classification with PyTorch

### Lecture Notebook:

A 2-D non-linear dataset that forms two concentric circles is used in this lecture. The first model that is used to make predictions is a neural network with two linear layers. The loss was high and the accuracy was low for this model.

![Making predictions on non-linear data with model 0.][lecture_02_00]

*The predictions on 2-D concentric circles dataset of a neural network with 2 linear layers.*

The model was then improved with more linear layers, more hidden units, and more epochs to train on. The results for this "improved" model were also poor. We then tested the model to see if it can make predictions on straight-line data like lecture 01. The improved model had low loss and high accuracy with linear data.

Knowing we needed a model with non-linearity, a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function was added to create a new model that is able to make predictions on non-linear data. This new non-linear model had a test loss of around 0.035 and a test accuracy of 100%.


![Making predictions on non-linear data with model 1.][lecture_02_01]

*The predictions on 2-D concentric circles dataset of a neural network using the non-linear ReLU activation function.*

In the next part of the lecture, we replicated, with code, non-linear activation functions.

We then built a multi-class classification neural network model (linear->ReLU-Linear->ReLU-Linear). We used [SKLearn's make_blobs dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) to make four isotropic Gaussian blobs to classify. This architecture had a testing loss of 0.0266 and over 99.50% test accuracy.

![Decision boundaries on blob dataset][lecture_02_02]

*Decision boundaries on training and testing blob datgasets.*

### Exercise Notebook:

Used [SKLearn's make_moons dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) where I created 2 moons to attempt to make a decision boundary on. The architecture of the neural network model that I used consisted of 3 hidden linear layers using ReLU activation functions. This model had a test loss of 0.004 and a test accuracy of 100%.

![Decision boundaries on make moons dataset][exercise_02_00]

*Decision boundary on make moons dataset.*

I then created a spiral dataset using a [function](https://cs231n.github.io/neural-networks-case-study/) from Stanford's [CS231n: Deep Learning for Computer Vision course](http://cs231n.stanford.edu/). I first tried a softmax regression model which acheived a test accuracy of 50% with a loss of 0.759.

![Decision bondaries on spiral dataset using softmax regression model][exercise_02_01]

*Decision boundaries on spiral dataset using a softmax regression model.*

To improve accuracy, I created a neural network with one hidden linear layer using a ReLU activation function. Over 100 epoches, this new model achieved 100% accuracy.

![Decision bondaries on spiral dataset using model with one hidden layer][exercise_02_02]

*Decision boundaries on spiral dataset using a neural network model with one hidden layer.*

</details>

<details>

<summary><b>Section 03: PyTorch Computer Vision</b></summary>

## Section 03: PyTorch Computer Vision

### Lecture Notebook:

Made classification predictions on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. I was introduced to PyTorch [DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and timing the training of my model's.

| Model               | Decription                                                                                                                                           |
| ---                 | ---                                                                                                                                                  |
| FashionMNISTModelV0 | Neural Network with 2 linear layers and trained on the CPU                                                                                           |
| FashionMNISTModelV1 | Similar to V0 but trained on the GPU and uses ReLU activation functions in between linear layers to lear non linearity                               |
| FashionMNISTModelV2 | CNN model architecture that replicates TinyVGG used on the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website and trained on the GPU |

Below is a comparison of model results. All models trained for 3 epochs.

| Model               | loss     | accuracy   | training time  |
| ---                 | ---      | ---        |  ---           |
| FashionMNISTModelV0 | 0.476639 | 83.426518  | 35.320797	   |
| FashionMNISTModelV1 | 0.685001 | 75.019968  | 32.601134      |
| FashionMNISTModelV2 | 0.321644 | 88.448482  | 36.510819      |

We then used matplotlib to print out images of random predictions and we were introduced to confusion matrices and saving and loading the best performing model.

### Exercise Notebook:

I worked with the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. I used a CNN with the Tiny VGG architecture. Before training I visualized the data. On the test data this model had a test accuracy of 87.39% with a train time 73.076 seconds. I then plotted a confusion matrix to compare the model's predictions to the truth labels. I then visualized wrong predictions and inferred why they may have been classified wrong, which I think this is due to error in data where some images in certain classes look too similar to images in other classes. For example, it is hard to distinguish between images of sneakers and images of ankle boots or images of shirts and images of coats, etc.

![Visual of correct MNIST classifications][exercise_03_00]

*Visual of Tiny VGG predictions on MNIST*

![Confustion matrix for MNIST classifications][exercise_03_01]

*Confustion matrix for MNIST classifications*

![Visual of incorrect MNIST classifications][exercise_03_02]

*Visual of incorrect Tiny VGG predictions on MNIST*

</details>

<details>

<summary><b>Section 04: PyTorch Custom Datasets</b></summary>

## Section 04: PyTorch Custom Datasets

### Lecture Notebook:

Used a custom dataset with a subset of the [Food101](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdata.vision.ee.ethz.ch%2Fcvl%2Fdatasets_extra%2Ffood-101%2F) dataset. This is a datset with pizza, setak, and sushi images. I learned how to create custom datasets, utilize `torchvision.transforms` for data transformation, employ `ImageFolder` and a custom `Dataset` class for loading image data, and apply data augmentation. Additionally we built two models, neither particularly great. One model used the [TinyVGG](https://poloclub.github.io/cnn-explainer/) architecture, and the other also used TinyVGG architecture but augmented the data. I then explored loss curves, compared model results, and made a prediction on a custom image.

![Comparing model results][lecture_04_00]

*A comparison of the results of two different models.*

</details>

[exercise_01]: /images/Exercise_01.jpg
[lecture_02_00]: /images/Lecture_02_00.jpg
[lecture_02_01]: /images/Lecture_02_01.jpg
[lecture_02_02]: /images/Lecture_02_02.jpg
[exercise_02_00]: /images/Exercise_02_00.jpg
[exercise_02_01]: /images/Exercise_02_01.jpg
[exercise_02_02]: /images/Exercise_02_02.jpg
[exercise_03_00]: /images/Exercise_03_00.jpg
[exercise_03_01]: /images/Exercise_03_01.jpg
[exercise_03_02]: /images/Exercise_03_02.jpg
[lecture_04_00]: /images/Lecture_04_00.jpg
