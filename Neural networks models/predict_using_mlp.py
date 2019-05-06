"""
predict_using_mlp.py
--------------------
This code uses a trained MLP model to predict
time series based on a given input seed.

Requires:
    moehlis_test_data_###.mat - a file containing time series that were not used
    for training. This file contains a 3D array of size (nTS, nTP, 9) where
    nTS - number of time series
    nTP - number of time points

Creates:
    multiple seq_##.mat files in the given path, each file containing a single
    test and predicted time series for all 9 amplitudes.

The code has been used for the results in:
    "Predictions of turbulent shear flows using deep neural networks"
    P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
    Physical Review Fluids (accepted)

"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import load_model

################################################################################
# Settings
################################################################################

# The indices of time series in the test data to be predicted.
# Both arguments have to be less than (or equal, for the second argument)
# to the number of time series in the data file.
seqNoList = range(0, 10)

# Prediction order (p), length of the sequence used for prediction of the next
# state.
seqLen = 500

# Path for the Keras model
model = load_model('MLP3_t100.h5')

# For the test data filename, include full path if in a different location
dataFilename = "moehlis_test_data_100.mat"

# File path and name for predicted time series
# Create a folder "MLP3_t100_ps" for the following line to be valid
saveFilename = "./MLP3_t100_ps/series_"

################################################################################
# Extract the data
dataStruct = sio.loadmat(dataFilename)
A = dataStruct['data']

# Make predictions
for seqNo in seqNoList:
    print(seqNo + 1)

    # Select a test sequence
    testSeq = A[seqNo]

    # Copy the first seqLen entries of the test sequence into the predicted
    # sequence. This will act as the input seed.
    predSeq = testSeq[:seqLen].reshape(1, -1)

    # Predict the sequence one new state at a time, and append to the list
    for i in np.arange(0, testSeq.shape[1]*(testSeq.shape[0]-seqLen),
                       testSeq.shape[1]):
        nextState = model.predict(predSeq[:, i:i+seqLen*testSeq.shape[1]],
                                  verbose=0)
        predSeq = np.concatenate((predSeq, nextState), axis=1)

    # Reshape the target sequence to a 2D array with 9 columns
    predSeq = predSeq.reshape(-1, 9)

    # Save the test and predicted sequence
    sio.savemat(saveFilename + '%d' % (seqNo + 1),
                {'testSeq':testSeq, 'predSeq':predSeq})
