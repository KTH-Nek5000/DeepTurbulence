"""
train_mlp_model.py
------------------
Requires:
    moehlis_data_###.mat obtained from the 9-mode ODE model
    The file contains a single 3D array of the size (nTS, nTP, 9), where
    nTS - number of time series
    nTP - number of time points

Creates:
    ****.h5 model file for Python Keras that contains the trained MLP model
    including weights and biases.

    ****_loss.mat file that contains the training and validation losses

The code has been used for the results in:
    "Predictions of turbulent shear flows using deep neural networks"
    P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
    Physical Review Fluids (accepted)

Note:
    This file may also require the created files ****.h5 and ****_loss.mat
    if the setting train_more_epochs is True.
"""
import numpy as np
import scipy.io as sio
from keras.models import Sequential, load_model
from keras.layers import Dense
import timeit

################################################################################
# Training settings
################################################################################

# Prediction order (p), length of the sequence used for prediction of the next
# state.
seqLen = 500

# The MLP architecture in an array
# For three hidden layers of 45 units each, this array is [45, 45, 45]
hiddenLayers = [90, 90, 90, 90]
arch = [seqLen*9] + hiddenLayers + [9]
print("\nNeural Network Architecture : %s" % arch)

# To train a new model from scratch, this setting is False.
# To train a pre-existing model for more epochs, this setting is True.
train_more_epochs = False

# Number of epochs for training
nbEpochs = 10

# For the data filename, include full path if in a different location
dataFilename = "moehlis_data_100.mat"

# Filename to save the model and losses
saveFilename = "MLP3_t100"

################################################################################
# Build or load the Keras model
if train_more_epochs:
    model = load_model(saveFilename + '.h5')

else:
    model = Sequential()
    # Add the hidden layers
    for i in range(len(hiddenLayers)):
        model.add(Dense(arch[i+1], input_dim=arch[i], activation='tanh',
                        kernel_initializer='glorot_normal'))

    # Add the output layer
    model.add(Dense(arch[-1], activation='tanh',
                    kernel_initializer='glorot_normal'))

    model.compile(optimizer='adam', loss='mean_squared_error')

################################################################################
# Load file and train the network
dataStruct = sio.loadmat(dataFilename)
A = dataStruct['data']

# The number of training samples
nSamples = A.shape[0]*(A.shape[1]-seqLen)

# Initialize the training inputs and outputs using empty arrays
X = np.empty([nSamples, seqLen*A.shape[2]])
Y = np.empty([nSamples, A.shape[2]])

# Fill the input and output arrays with data
k = 0
for i in np.arange(A.shape[0]):
    for j in np.arange(A.shape[1]-seqLen):
        X[k] = A[i, j:j+seqLen].reshape(1,-1)
        Y[k] = A[i, j+seqLen].reshape(1,-1)
        k = k + 1

# Timer
tStart = timeit.default_timer()

# Training
score = model.fit(X, Y, batch_size=32, epochs=nbEpochs,
                  verbose=1, validation_split=0.20, shuffle=True)

tStop = timeit.default_timer()

# Save losses
lossHistory = score.history['loss']
valLossHistory = score.history['val_loss']

########################################################################
# Save the results
########################################################################
# Keras model
model.save(saveFilename + '.h5')

# For the loss file, append the previous training history to the new history
if train_more_epochs:
    prevLossHistory = sio.loadmat(saveFilename + '_loss.mat')
    lossHistory = np.concatenate((prevLossHistory['lossHistory'].flatten(),
                                  lossHistory))
    valLossHistory = \
        np.concatenate((prevLossHistory['valLossHistory'].flatten(),
                        valLossHistory))

# Save the losses
sio.savemat(saveFilename + '_loss',
    {'lossHistory':lossHistory, 'valLossHistory':valLossHistory})

print("\nThe training time is %f sec" % (tStop - tStart))
