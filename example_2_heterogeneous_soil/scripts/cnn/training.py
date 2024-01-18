# Import the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os

from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, losses, metrics
from scipy.io import loadmat, savemat

# Change fonts and specify font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
FONT_SIZE = 12

# Get displacement and force data
def get_data(fileDirectory):

    # Get the training displacement data
    displacementTrainingData = loadmat(fileDirectory + '/displacementTrainingData.mat')
    displacementTrainingData = np.array(tuple(displacementTrainingData['data']))
    
    # Get the training force data
    forceTrainingData = loadmat(fileDirectory + '/forceTrainingData.mat')
    forceTrainingData = np.array(tuple(forceTrainingData['data']))
    
    # Get the validation displacement data
    displacementValidationData = loadmat(fileDirectory + '/displacementValidationData.mat')
    displacementValidationData = np.array(tuple(displacementValidationData['data']))
    
    # Get the validation force data
    forceValidationData = loadmat(fileDirectory + '/forceValidationData.mat')
    forceValidationData = np.array(tuple(forceValidationData['data']))
    
    # Get the test displacement data
    displacementTestData = loadmat(fileDirectory + '/displacementTestData.mat')
    displacementTestData = np.array(tuple(displacementTestData['data']))
    
    # Get the test force data
    forceTestData = loadmat(fileDirectory + '/forceTestData.mat')
    forceTestData = np.array(tuple(forceTestData['data']))

    return displacementTrainingData, forceTrainingData, displacementValidationData, forceValidationData, displacementTestData, forceTestData

# Define function to normalize the input and output dataset
def normalize_data(dataUsedToNormalize, dataToBeNnormalized, typeOfDataSubset):

    dataMean = np.mean(dataUsedToNormalize)
    dataRange = np.max(dataUsedToNormalize) - np.min(dataUsedToNormalize)

    normalizedData = (dataToBeNnormalized - dataMean) / dataRange
    normalizingParameters = np.array(list([dataMean, dataRange]))

    print('Normalized %s dataset.' %(typeOfDataSubset))

    return normalizedData, normalizingParameters

# Define function to normalize the input and output dataset
def renormalize_data(dataUsedToRenormalize, dataToBeRennormalized, typeOfDataSubset):

    dataMean = np.mean(dataUsedToRenormalize)
    dataRange = np.max(dataUsedToRenormalize) - np.min(dataUsedToRenormalize)

    renormalizedData = (dataToBeRennormalized * dataRange) + dataMean

    print('Renormalized %s dataset.' %(typeOfDataSubset))

    return renormalizedData

class cnn():

    def __init__(self, inputShape):
        
        self.inputShape = inputShape
    
    def build_model(self):

        model = tf.keras.Sequential([
            
            # Input Layer
            layers.Input(shape=self.inputShape),
            
            # Hidden Layers
            layers.Conv1D(filters=512, kernel_size=40, padding='same', kernel_initializer='glorot_normal', activation='LeakyReLU'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=352, kernel_size=50, padding='same', kernel_initializer='he_uniform', activation='LeakyReLU'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=64, kernel_size=40, padding='same', kernel_initializer='he_normal', activation='LeakyReLU'),
            layers.BatchNormalization(),
            
            # Output Layer
            layers.Conv1D(filters=126, kernel_size=55, padding='same', kernel_initializer='glorot_normal')])
        
        return model

class plots:
    
    def __init__(self, history, fileDirectory):

        self.history = history
        self.fileDirectory = fileDirectory
        self.fontsize = 15

    def loss(self):

        loss_name = list(self.history.history.keys())[0]

        # Training and Validation
        loss = self.history.history[loss_name]
        valLoss = self.history.history['val_' + loss_name]

        lossPlot = plt.figure()
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, 'bo--', label = 'Training Loss', markersize = 2)
        plt.plot(epochs, valLoss, 'go--', label = 'Validation Loss', markersize = 2)
        plt.title('Training and Validation Loss', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Loss', fontsize=self.fontsize)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=self.fontsize)
        plt.savefig(self.fileDirectory + '/loss.pdf', bbox_inches='tight')

        return lossPlot

    def evaluation_metric(self):

        metric_name = list(self.history.history.keys())[1]
        
        # Training and Validation
        metric = self.history.history[metric_name]
        valMetric = self.history.history['val_' + metric_name]

        metricPlot = plt.figure()
        epochs = range(1, len(metric)+1)
        plt.plot(epochs, metric, 'bo--', label = 'Training Metric', markersize = 2)
        plt.plot(epochs, valMetric, 'go--', label = 'Validation Metric', markersize = 2)
        plt.title('Training and Validation Evaluation Metric', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Evaluation Metric', fontsize=self.fontsize)
        plt.legend(['Training Metric', 'Validation Metric'], fontsize=self.fontsize)
        plt.savefig(self.fileDirectory + '/evaluation_metric.pdf', bbox_inches='tight')

        return metricPlot

def swap_sensor_timestep(data):

    newData = np.zeros((data.shape[0], data.shape[2], data.shape[1]))

    for nSamples in range(newData.shape[0]):
        for nSensors in range(newData.shape[1]):
            for nTimesteps in range(newData.shape[2]):

                newData[nSamples, nSensors, nTimesteps] = data[nSamples, nTimesteps, nSensors]
    
    return newData

# Define function to save the dataset
def save_data(data, fileName, file_directory_to_save_data):
    
    savemat(file_directory_to_save_data + '/' + fileName, {'data': data})
        
    print('Saved ' + fileName)

def main():

    # Define file structure heirarchy
    mainFileDirectory = 'YOUR_FILE_DIRECTORY'
    dividedDataFileDirectory = mainFileDirectory + '/dividedDataset'
    fileDirectoryToSaveResults = mainFileDirectory + '/cnn/trainingResults'

    # Get displacement and void data
    displacementTrainingData, forceTrainingData, displacementValidationData, forceValidationData, displacementTestData, forceTestData = get_data(dividedDataFileDirectory)

    # Get the number of timesteps from the initial data
    with open(dividedDataFileDirectory + '/metadataInitialData.pkl', 'rb') as handle:
        metaData = pickle.load(handle)
    
    # Normalize the displacement training datasets
    displacementValidationData = normalize_data(dataUsedToNormalize = displacementTrainingData, dataToBeNnormalized = displacementValidationData, typeOfDataSubset = 'Displacement Validation')[0]
    displacementTestData = normalize_data(dataUsedToNormalize = displacementTrainingData, dataToBeNnormalized = displacementTestData, typeOfDataSubset = 'Displacement Test')[0]
    displacementTrainingData, normalizingDisplacementParameters = normalize_data(dataUsedToNormalize = displacementTrainingData, dataToBeNnormalized = displacementTrainingData, typeOfDataSubset = 'Displacement Training')

    unnormalizedForceTraining = forceTrainingData

    # Normalize the force training datasets
    forceValidationData = normalize_data(dataUsedToNormalize = forceTrainingData, dataToBeNnormalized = forceValidationData, typeOfDataSubset = 'Force Validation')[0]
    forceTestData = normalize_data(dataUsedToNormalize = forceTrainingData, dataToBeNnormalized = forceTestData, typeOfDataSubset = 'Force Test')[0]
    forceTrainingData, normalizingDisplacementParameters = normalize_data(dataUsedToNormalize = forceTrainingData, dataToBeNnormalized = forceTrainingData, typeOfDataSubset = 'Force Training')

    epochs = 1
    model = cnn(displacementTrainingData.shape[1:]).build_model()
    model.compile(optimizer=optimizers.Adamax(learning_rate=0.001), loss='mae', metrics='mse')

    model.summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(displacementTrainingData, forceTrainingData, epochs=epochs, batch_size=16, callbacks=[stop_early], validation_data=(displacementValidationData, forceValidationData))

    # Save the trained model
    model.save(fileDirectoryToSaveResults + '/trainedModel.h5')
    print('Saved trainedModel.h5 to %s.' %fileDirectoryToSaveResults)
    
    plot = plots(history, fileDirectoryToSaveResults)
    lossPlot = plot.loss()
    evaluationMetricPlot = plot.evaluation_metric()

if __name__ == '__main__':

    os.system('clear')
    main()
