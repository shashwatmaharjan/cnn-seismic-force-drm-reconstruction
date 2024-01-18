# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random

from mat73 import loadmat
from scipy.io import savemat
from sklearn.model_selection import train_test_split

# Define necessary functions
# Get the generated raw data
def get_data(fileDirectory, fileName):

    # Load the data
    data = loadmat(fileDirectory + '/' + fileName)
    data = np.array(tuple(data.values()))

    # Remove the unnecessary dimension
    data = np.squeeze(data)

    return data


def data_skip(data, timeSkip):

    # Take timesteps by skipping a few timesteps
    return data[:, :, 1::timeSkip]


# Define function to divide dataset into training, validation, and test dataset
def dataset_divider(displacementData, forceData):

    # Split training data into 75%, 15%, and 10% respectively
    displacementTrainingData, displacementTestData, forceTrainingData, forceTestData = train_test_split(displacementData, forceData, test_size=0.1, random_state=1)
    displacementTrainingData, displacementValidationData, forceTrainingData, forceValidationData = train_test_split(displacementTrainingData, forceTrainingData, test_size=0.1, random_state=1)

    return displacementTrainingData, forceTrainingData, displacementValidationData, forceValidationData, displacementTestData, forceTestData 

# Define a function that switches the position of sensors and timesteps for training
def swap_sensor_timestep_dimension(data):

    # Get the number of samples, sensors, and timesteps
    numSamples = data.shape[0]
    numSensors = data.shape[1]
    numTimestep = data.shape[2]

    # Create an empty array to store the swapped data
    swappedData = np.zeros((numSamples, numTimestep, numSensors))

    # Swap the position of sensors and timesteps
    for nSamples in range(numSamples):

        for nSensors in range(numSensors):

            for nTimesteps in range(numTimestep):

                swappedData[nSamples, nTimesteps, nSensors] = data[nSamples, nSensors, nTimesteps]
    
    return swappedData


# Define function to save the dataset
def save_data(data, fileName, file_directory_to_save_data):

    # Save the data
    # Check if the file is a pickle file or a mat file
    if fileName[-4:] == '.pkl':

        with open(file_directory_to_save_data + fileName, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif fileName[-4:] == '.mat':
        savemat(file_directory_to_save_data + '/' + fileName, {'data': data})
        
    print('Saved ' + fileName)


# Define main function
def main():

    print('Running dataAssembler.py')

    # Assign the file directory as a string to a variable for re-usability
    mainFileDirectory = 'YOUR_FILE_DIRECTORY'
    generatedDataFileDirectory = mainFileDirectory + '/generateDataset'
    fileDirectoryToSaveData = mainFileDirectory + '/dividedDataset'

    # Get displacement and force data
    displacementData = get_data(generatedDataFileDirectory, 'u_history.mat')
    forceData = get_data(generatedDataFileDirectory, 'F_history.mat')

    # To provide as close of a one-to-one ratio, take timesteps by skipping a few timesteps
    # displacementData = data_skip(displacementData, 3)
    # forceData = data_skip(forceData, 3)

    # Save the meta data containing information about the displacement and force dataset
    metadata = {'displacement': {'samples': np.shape(displacementData)[0], 'sensors': np.shape(displacementData)[1], 'timesteps': np.shape(displacementData)[2]},
                'force': {'samples': np.shape(forceData)[0], 'sensors': np.shape(forceData)[1], 'timesteps': np.shape(forceData)[2]}}
    print('\nSaving metadata...')
    save_data(metadata, '/metadataInitialData.pkl', fileDirectoryToSaveData)

    # Divide the data into training, validation, and test dataset
    displacementTrainingData, forceTrainingData, displacementValidationData, forceValidationData, displacementTestData, forceTestData = dataset_divider(displacementData, forceData)

    # Swap the position of sensors and timesteps for training
    displacementTrainingData = swap_sensor_timestep_dimension(displacementTrainingData)
    displacementValidationData = swap_sensor_timestep_dimension(displacementValidationData)
    displacementTestData = swap_sensor_timestep_dimension(displacementTestData)

    # Swap the position of sensors and timesteps for training
    forceTrainingData = swap_sensor_timestep_dimension(forceTrainingData)
    forceValidationData = swap_sensor_timestep_dimension(forceValidationData)
    forceTestData = swap_sensor_timestep_dimension(forceTestData)

    # Save the dataset
    save_data(displacementTrainingData, 'displacementTrainingData.mat', fileDirectoryToSaveData)
    save_data(forceTrainingData, 'forceTrainingData.mat', fileDirectoryToSaveData)
    save_data(displacementValidationData, 'displacementValidationData.mat', fileDirectoryToSaveData)
    save_data(forceValidationData, 'forceValidationData.mat', fileDirectoryToSaveData)
    save_data(displacementTestData, 'displacementTestData.mat', fileDirectoryToSaveData)
    save_data(forceTestData, 'forceTestData.mat', fileDirectoryToSaveData)

if __name__ == '__main__':

    os.system('clear')

    main()
