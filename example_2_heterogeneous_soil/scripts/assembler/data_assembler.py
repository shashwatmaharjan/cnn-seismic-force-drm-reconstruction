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
# Function to get the generated raw data
def get_data(file_directory, file_name):

    # Load the data
    data = loadmat(file_directory + '/' + file_name)
    data = np.array(tuple(data.values()))

    data = np.squeeze(data)

    return data

# Function to skip timesteps
def data_skip(data, time_skip):

    # Take timesteps by skipping a few timesteps
    return data[:, :, 1::time_skip]

# Function to divide dataset into training, validation, and test dataset
def dataset_divider(displacement_data, force_data):

    # Split training data into 75%, 15%, and 10% respectively
    displacement_training_data, displacement_test_data, force_training_data, force_test_data = train_test_split(displacement_data, force_data, test_size=0.1, random_state=1)
    displacement_training_data, displacement_validation_data, force_training_data, force_validation_data = train_test_split(displacement_training_data, force_training_data, test_size=0.1, random_state=1)

    return displacement_training_data, force_training_data, displacement_validation_data, force_validation_data, displacement_test_data, force_test_data 

# Define a function that switches the position of sensors and timesteps for training
def swap_sensor_timestep_dimension(data):

    # Get the number of samples, sensors, and timesteps
    num_samples = data.shape[0]
    num_sensors = data.shape[1]
    num_timestep = data.shape[2]

    # Create an empty array to store the swapped data
    swapped_data = np.zeros((num_samples, num_timestep, num_sensors))

    # Swap the position of sensors and timesteps
    for n_sample in range(num_samples):

        for n_sensor in range(num_sensors):

            for n_timestep in range(num_timestep):

                swapped_data[n_sample, n_timestep, n_sensor] = data[n_sample, n_sensor, n_timestep]
    
    return swapped_data


# Define function to save the dataset
def save_data(data, file_name, file_directory_to_save_data):

    if file_name[-4:] == '.pkl':

        with open(file_directory_to_save_data + file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif file_name[-4:] == '.mat':
        savemat(file_directory_to_save_data + '/' + file_name, {'data': data})
        
    print('Saved ' + file_name)


# Define main function
def main():

    print('Running dataAssembler.py')

    # Define the example name
    example = 'example_1_homogeneous_soil'

    # Assign the file directory as a string to a variable for re-usability
    main_file_directory = os.getcwd()
    generated_data_file_directory = os.path.join(main_file_directory, example, 'data', 'generated')
    file_directory_to_save_data = os.path.join(main_file_directory, example, 'data', 'assembled')

    # Get displacement and force data
    displacement_data = get_data(generated_data_file_directory, 'u_history.mat')
    force_data = get_data(generated_data_file_directory, 'F_history.mat')

    # To provide as close of a one-to-one ratio, take timesteps by skipping a few timesteps
    # displacement_data = data_skip(displacement_data, 3)
    # force_data = data_skip(force_data, 3)

    # Save the meta data containing information about the displacement and force dataset
    metadata = {'displacement': {'samples': np.shape(displacement_data)[0], 'sensors': np.shape(displacement_data)[1], 'timesteps': np.shape(displacement_data)[2]},
                'force': {'samples': np.shape(force_data)[0], 'sensors': np.shape(force_data)[1], 'timesteps': np.shape(force_data)[2]}}
    print('\nSaving metadata...')
    save_data(metadata, '/metadataInitialData.pkl', file_directory_to_save_data)

    # Divide the data into training, validation, and test dataset
    displacement_training_data, force_training_data, displacement_validation_data, force_validation_data, displacement_test_data, force_test_data = dataset_divider(displacement_data, force_data)

    displacement_training_data = swap_sensor_timestep_dimension(displacement_training_data)
    displacement_validation_data = swap_sensor_timestep_dimension(displacement_validation_data)
    displacement_test_data = swap_sensor_timestep_dimension(displacement_test_data)

    force_training_data = swap_sensor_timestep_dimension(force_training_data)
    force_validation_data = swap_sensor_timestep_dimension(force_validation_data)
    force_test_data = swap_sensor_timestep_dimension(force_test_data)

    save_data(displacement_training_data, 'displacement_training_data.mat', file_directory_to_save_data)
    save_data(force_training_data, 'force_training_data.mat', file_directory_to_save_data)
    save_data(displacement_validation_data, 'displacement_validation_data.mat', file_directory_to_save_data)
    save_data(force_validation_data, 'force_validation_data.mat', file_directory_to_save_data)
    save_data(displacement_test_data, 'displacement_test_data.mat', file_directory_to_save_data)
    save_data(force_test_data, 'force_test_data.mat', file_directory_to_save_data)


if __name__ == '__main__':

    os.system('clear')

    main()
