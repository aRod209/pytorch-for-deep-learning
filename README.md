# pytorch-for-deep-learning
Notebooks of lessons and exercises for the course "Learn PyTorch for Deep Learning: Zero to Mastery"

## Section 00: PyTorch Fundamentals
**Lecture Notebook:** Going over PyTorch fundamentals such as creating tensors in different ways, finding information about tensors, tensor operations for tensor manipulation, matrix operations, and comparaing PyTorch tensors and NumPy arrays.

**Exercise Notebook:** Simple exercises of creating PyTorch tensors and manipulating them.

## Section 01: PyTorch Workflow
**Lecture Notebook:** Going over a PyTorch end-to-end workflow. We start by preparing and loading data, then we build a linear model and we train the data on theh linear model. While training thhe data we plot the curves of the training loss and testing loss. We then go over saving and loading the training model.

**Exercise Notebook:** Replicate the PyTorch workflow using a straight-line dataset. A linear model is created by subclassing `nn.Module`. We create a loss function with `nn.L1Loss` and an optimizer using `torch.optim.SGD`. I train the model and then I made predictions using the test data. Lastly, I saved the trained model's `state_dict` to a file.

![Making predictions with the training data and testing data of a linear dataset with a linear model.][exercise_01]

*Making predictions with the training data and testing data of a linear dataset with a linear model.*

## Section 02: Neural Network classification with PyTorch
**Lecture Notebook:** A 2-D non-linear dataset that forms two concentric circles is used in this lecture. The first model that is used to make predictions is a neural network with two linear layers. The loss was high and the accuracy was low for this model.

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

[exercise_01]: /images/Exercise_01.jpg
[lecture_02_00]: /images/Lecture_02_00.jpg
[lecture_02_01]: /images/Lecture_02_01.jpg
[lecture_02_02]: /images/Lecture_02_02.jpg