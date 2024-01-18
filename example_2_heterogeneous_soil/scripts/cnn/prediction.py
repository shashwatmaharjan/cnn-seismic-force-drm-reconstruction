# Import the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os

from tensorflow import keras
from tensorflow.keras import models
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
    
    # Get the test displacement data
    displacementTestData = loadmat(fileDirectory + '/displacementTestData.mat')
    displacementTestData = np.array(tuple(displacementTestData['data']))
    
    # Get the test force data
    forceTestData = loadmat(fileDirectory + '/forceTestData.mat')
    forceTestData = np.array(tuple(forceTestData['data']))

    return displacementTrainingData, forceTrainingData, displacementTestData, forceTestData

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

def error_calculator(target, prediction):

    numerator = np.sum(np.sum((np.abs(target - prediction))**2, axis=1), axis=1)
    denominator = np.sum(np.sum((np.abs(target))**2, axis=1), axis=1)
    error = numerator/denominator*100

    return error

def five_statistic(error):

    sortedError = np.sort(error)

    # Minimum
    minError = np.min(error)

    # 25th percentile
    q1Location = int(len(sortedError)/4)
    q1Error = sortedError[q1Location]

    # 50th percentile
    medianLocation = int(len(sortedError)/2)
    medianError = sortedError[medianLocation]

    # 75th percentile 
    q3Location = 3*int(len(sortedError)/4)
    q3Error = sortedError[q3Location]

    # Maximum
    maxError = np.max(error)

    return minError, q1Error, medianError, q3Error, maxError

def five_statistic_location(error):

    sortedError = np.sort(error)

    # Minimum
    minErrorLocation = np.where(error == np.min(error))[0]

    # 25th percentile
    q1Location = int(len(sortedError)/4)

    # 50th percentile
    medianLocation = int(len(sortedError)/2)

    # 75th percentile 
    q3Location = 3*int(len(sortedError)/4)

    # Maximum
    maxErrorLocation = np.where(error == np.max(error))[0]

    return minErrorLocation, q1Location, medianLocation, q3Location, maxErrorLocation

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
    trainedModelDirectory = mainFileDirectory + '/cnn/trainingResults'
    fileDirectoryToSaveResults = mainFileDirectory + '/cnn/predictionResults'

    # Get displacement and void data
    displacementTrainingData, forceTrainingData, displacementTestData, forceTestData = get_data(dividedDataFileDirectory)
    
    # Normalize the force training datasets
    forceTestData = normalize_data(dataUsedToNormalize = forceTrainingData, dataToBeNnormalized = forceTestData, typeOfDataSubset = 'Force Test')[0]
    forceTrainingData, normalizingDisplacementParameters = normalize_data(dataUsedToNormalize = forceTrainingData, dataToBeNnormalized = forceTrainingData, typeOfDataSubset = 'Force Training')

    # Save the trained model
    model = models.load_model(trainedModelDirectory + '/trainedModel.h5')

    model.summary()

    prediction = model.predict(displacementTestData)

    unnormalizedForceTraining = forceTrainingData
    
    # Renormalize target and prediction
    prediction = renormalize_data(dataUsedToRenormalize = unnormalizedForceTraining, dataToBeRennormalized = prediction, typeOfDataSubset = 'Prediction')
    target = renormalize_data(dataUsedToRenormalize = unnormalizedForceTraining, dataToBeRennormalized = forceTestData, typeOfDataSubset = 'Target')

    # Get error metric
    error = error_calculator(target, prediction)

    prediction = swap_sensor_timestep(prediction)
    target = swap_sensor_timestep(target)

    # Five statistics
    minError, q1Error, medianError, q3Error, maxError = five_statistic(error)
    minErrorLocation, q1ErrorLocation, medianErrorLocation, q3ErrorLocation, maxErrorLocation = five_statistic_location(error)

    error_indices = {'min': minErrorLocation, 'max': q1ErrorLocation, 'q1': medianErrorLocation, 'median': q3ErrorLocation, 'q3': maxErrorLocation}
    savemat(fileDirectoryToSaveResults + '/error_indices.mat', error_indices)

    prediction = swap_sensor_timestep(prediction)
    target = swap_sensor_timestep(target)

    save_data(prediction, 'prediction.mat', fileDirectoryToSaveResults)
    save_data(target, 'target.mat', fileDirectoryToSaveResults)

if __name__ == '__main__':

    os.system('clear')
    main()
