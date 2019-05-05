# Deepturb

## Introduction

## Data generation

The MATLAB scripts in [Data generator (Moehlis model)]() are used to generate and visualize the training, validation and test datasets for the neural networks architectures. In particular:
* [moehlis_model_script.m]() generates a single time series
* [moehlis_data_gen.m]() generates a user-defined number of series and gather them in a single MATLAB file
* [plot_amplitudes.m]() and [visualize_fields.m]() can be used to check the amplitudes of the different modes

## Training neural networks

Once the time series dataset is created, neural networks can be trained on this dataset. Two different architectures can be chosen, multilayer perceptron (MLP) networks or Long short-term memory (LSTM) networks. The trained model is saved in format *h5*.

## Prediction of new time series

The trained model can be used to predict new timeseries, based on a initial sequence of *p* elements.
It is possible to use one of the already-trained models or train a new one from scratch.
