# pytorch-for-deep-learning
Notebooks of lessons and exercises for the course "Learn PyTorch for Deep Learning: Zero to Mastery"

## Section 00: PyTorch Fundamentals
**Lecture Notebook:** Going over PyTorch fundamentals such as creating tensors in different ways, finding information about tensors, tensor operations for tensor manipulation, maatrix operations, and comparaing PyTorch tensors and NumPy arrays.

**Exercise Notebook:** Simple exercises of creating PyTorch tensors and manipulating them.

## Section 01: PyTorch Workflow
**Lecture Notebook:** Going over a PyTorch end-to-end workflow. We start by preparing and loading data, then we build a linear model and we train the data on theh linear model. While training thhe data we plot the curves of the training loss and testing loss. We then go over saving and loading the training model.

**Exercise Notebook:** Replicate the PyTorch workflow using a straight-line dataset. A linear model is created by subclassing `nn.Module`. We create a loss function with `nn.L1Loss` and an optimizer using `torch.optim.SGD`. I train the model and then I made predictions using the test data. Lastly, I saved the trained model's `state_dict` to a file.
