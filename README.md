# UrbanSound8k Classification Project
## Prerequisites
### Downloading packages
*Python 3.9* : Run *pip install -r requirements.txt* in your pip env to install the necessary dependencies.
### Downloading the data
In order to use this project, you must download the UrbanSound8k data from the following URL [UrbanSound8k dataset](https://urbansounddataset.weebly.com/urbansound8k.html) and place *audio* in the data directory at the root of the project. Be sure to respect the existing directory structure and not rename the downloaded files.

Once the data has been placed correctly, you can start running the code.

## Introduction
The objective of this project is to build a classification model capable of distinguishing between different categories of urban sounds. We will use the UrbanSound8k dataset which contains recordings of urban sounds in different categories such as bird calls, bells, horns, etc. The goal is to build a model capable of classifying these sounds using the acoustic features extracted from each sound.

## Data preparation
We will start by loading the UrbanSound8k dataset and preparing the data for our classification model. This includes processing the audio signal to extract acoustic features, normalizing the data, and coding the output for the model.

## Building the model
Next, we will build 2 different type of neural network model for our classification. We will use the Keras library to build the model and compile the layers. We will also perform a cross validation to select the best hyperparameters for our model.

## Evaluation and analysis of results
Finally, we will evaluate our model on test data to assess its classification performance.

## Conclusion
This project allowed us to discover the different steps necessary to build a sound classification model. We learned how to extract acoustic features from audio recordings, how to build a neural network model and how to evaluate the performance of our model. We also gained practical experience using the Keras library to build and compile our model.