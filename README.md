# Advanced Deep Learning Coursework

This repository contains code I've written for the **Advanced Deep Learning** university course at the Chair of Machine Learning and Computer Vision, where I complete assignments focused on various computer vision tasks. Each assignment involves solving specific challenges using deep learning techniques and models.
This code isn't the exact code I submitted but ressembles the general way I've handled the task. Its structure follows the framework we've been given by the instructors but all the code itself is mine.

## Task: Image Classification of Simpsons Characters

### Overview
The first assignment focuses on training an image classification model (using pre-trained ResNet18 and ConvNextTiny as backbones) for a dataset of characters from *The Simpsons*. We've been provided the dataset but it can also be found [here](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset). The goal is to classify characters accurately based on the images provided in the dataset.

### Project Structure
- **dataset.py**: Functions to load and prepare the dataset using PyTorch, including setting up the necessary `DataLoader`. Does image augmentation like resizing and padding to 128x128 pixels, normalization and also splits the dataset into train and validation sets.
- **models.py**: Here I create my two models, one using the ResNet18 as backbone and the other using ConvNextTiny. For both I modify the last fully connected layer to adapt them to the number of classes that are in the dataset.
- **training.py**: Implement a train function that trains the provided model using the dataset and an optimizer. Uses mixed-precision training if a CUDA device is available. Evaluation function calculates the accuracies of all the classes, which are later averaged to get a total accuracy.
- **main.py**: Main function that starts the training. Chooses which device (CPU or GPU if available) is used, which model (ResNet18 or ConvNextTiny is trained for how many epochs (30) and which optimizer (adam with learning rate of 0.000)1 is chosen for learning.

### Results
My ResNet18 model has achieved a ~75% accuracy on evaluation and the ConvNextTiny model a ~90% accuracy.
