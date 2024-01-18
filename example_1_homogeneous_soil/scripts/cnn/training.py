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

# Define necessary functions
# Function to get the generated raw data
def get_data(file_directory):

    # Get the training displacement data
    displacement_training_data = loadmat(file_directory + '/displacement_training_data.mat')
    displacement_training_data = np.array(tuple(displacement_training_data['data']))
    
    # Get the training force data
    force_training_data = loadmat(file_directory + '/force_training_data.mat')
    force_training_data = np.array(tuple(force_training_data['data']))
    
    # Get the validation displacement data
    displacement_validation_data = loadmat(file_directory + '/displacement_validation_data.mat')
    displacement_validation_data = np.array(tuple(displacement_validation_data['data']))
    
    # Get the validation force data
    force_validation_data = loadmat(file_directory + '/force_validation_data.mat')
    force_validation_data = np.array(tuple(force_validation_data['data']))
    
    # Get the test displacement data
    displacement_test_data = loadmat(file_directory + '/displacement_test_data.mat')
    displacement_test_data = np.array(tuple(displacement_test_data['data']))
    
    # Get the test force data
    force_test_data = loadmat(file_directory + '/force_test_data.mat')
    force_test_data = np.array(tuple(force_test_data['data']))

    return displacement_training_data, force_training_data, displacement_validation_data, force_validation_data, displacement_test_data, force_test_data


# Function to normalize the input and output dataset
def normalize_data(data_used_to_normalize, data_to_be_normalized, type_of_data_subset):

    # Normalize the data
    data_mean = np.mean(data_used_to_normalize)
    data_range = np.max(data_used_to_normalize) - np.min(data_used_to_normalize)

    normalized_data = (data_to_be_normalized - data_mean) / data_range
    normalizing_parameters = np.array(list([data_mean, data_range]))

    print('Normalized %s dataset.' %(type_of_data_subset))

    return normalized_data, normalizing_parameters


# Function to renormalize the input and output dataset
def renormalize_data(data_used_to_renormalize, data_to_be_rennormalized, type_of_data_subset):

    # Renormalize the data
    data_mean = np.mean(data_used_to_renormalize)
    data_range = np.max(data_used_to_renormalize) - np.min(data_used_to_renormalize)

    renormalized_data = (data_to_be_rennormalized * data_range) + data_mean

    print('Renormalized %s dataset.' %(type_of_data_subset))

    return renormalized_data


# Class to build the CNN model
class CNN():

    def __init__(self, input_shape):
        
        self.input_shape = input_shape
    
    # Method to build the CNN model
    def build_model(self):

        model = tf.keras.Sequential([
            
            # Input Layer
            layers.Input(shape=self.input_shape),
            
            # Hidden Layers
            layers.Conv1D(filters=480, kernel_size=80, padding='same', kernel_initializer='he_uniform', activation='LeakyReLU'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=512, kernel_size=80, padding='same', kernel_initializer='he_uniform', activation='LeakyReLU'),
            layers.BatchNormalization(),
            
            # Output Layer
            layers.Conv1D(filters=126, kernel_size=90, padding='same', kernel_initializer='glorot_uniform')])
        
        return model


# Class to plot the training results
class plots:
    
    def __init__(self, history, file_directory):

        self.history = history
        self.file_directory = file_directory
        self.fontsize = 15

    # Method to plot the loss
    def loss(self):

        loss_name = list(self.history.history.keys())[0]

        # Training and Validation
        loss = self.history.history[loss_name]
        val_loss = self.history.history['val_' + loss_name]

        loss_plot = plt.figure()
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, 'bo--', label = 'Training Loss', markersize = 2)
        plt.plot(epochs, val_loss, 'go--', label = 'Validation Loss', markersize = 2)
        plt.title('Training and Validation Loss', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Loss', fontsize=self.fontsize)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=self.fontsize)
        plt.savefig(self.file_directory + '/loss.pdf', bbox_inches='tight')

        return loss_plot

    # Method to plot the evaluation metric
    def evaluation_metric(self):

        metric_name = list(self.history.history.keys())[1]
        
        # Training and Validation
        metric = self.history.history[metric_name]
        val_metric = self.history.history['val_' + metric_name]

        metricPlot = plt.figure()
        epochs = range(1, len(metric)+1)
        plt.plot(epochs, metric, 'bo--', label = 'Training Metric', markersize = 2)
        plt.plot(epochs, val_metric, 'go--', label = 'Validation Metric', markersize = 2)
        plt.title('Training and Validation Evaluation Metric', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Evaluation Metric', fontsize=self.fontsize)
        plt.legend(['Training Metric', 'Validation Metric'], fontsize=self.fontsize)
        plt.savefig(self.file_directory + '/evaluation_metric.pdf', bbox_inches='tight')

        return metricPlot


# Function to swap the sensor and timestep dimensions
def swap_sensor_timestep(data):

    # Create a new array to store the swapped data
    new_data = np.zeros((data.shape[0], data.shape[2], data.shape[1]))

    # Swap the sensor and timestep dimensions
    for nSamples in range(new_data.shape[0]):
        for nSensors in range(new_data.shape[1]):
            for nTimesteps in range(new_data.shape[2]):

                new_data[nSamples, nSensors, nTimesteps] = data[nSamples, nTimesteps, nSensors]
    
    return new_data


# Function to save the dataset
def save_data(data, fileName, file_directory_to_save_data):

    # Save the data
    savemat(file_directory_to_save_data + '/' + fileName, {'data': data})
        
    print('Saved ' + fileName)


def main():

    # Define file structure heirarchy
    main_file_directory = 'YOUR_FILE_DIRECTORY'
    divided_data_file_directory = main_file_directory + '/dividedDataset'
    file_directory_to_save_results = main_file_directory + '/cnn/trainingResults'

    # Get displacement and void data
    displacement_training_data, force_training_data, displacement_validation_data, force_validation_data, displacement_test_data, force_test_data = get_data(divided_data_file_directory)

    # Get the number of timesteps from the initial data
    with open(divided_data_file_directory + '/metadataInitialData.pkl', 'rb') as handle:

        meta_data = pickle.load(handle)
    
    # Normalize the displacement training datasets
    displacement_validation_data = normalize_data(data_used_to_normalize = displacement_training_data, data_to_be_normalized = displacement_validation_data, type_of_data_subset = 'Displacement Validation')[0]
    displacement_test_data = normalize_data(data_used_to_normalize = displacement_training_data, data_to_be_normalized = displacement_test_data, type_of_data_subset = 'Displacement Test')[0]
    displacement_training_data, normalizing_displacement_parameters = normalize_data(data_used_to_normalize = displacement_training_data, data_to_be_normalized = displacement_training_data, type_of_data_subset = 'Displacement Training')

    # Set the unnormalized force training dataset separately
    unnormalized_force_training = force_training_data

    # Normalize the force training datasets
    force_validation_data = normalize_data(data_used_to_normalize = force_training_data, data_to_be_normalized = force_validation_data, type_of_data_subset = 'Force Validation')[0]
    force_test_data = normalize_data(data_used_to_normalize = force_training_data, data_to_be_normalized = force_test_data, type_of_data_subset = 'Force Test')[0]
    force_training_data, normalizing_displacement_parameters = normalize_data(data_used_to_normalize = force_training_data, data_to_be_normalized = force_training_data, type_of_data_subset = 'Force Training')

    # Build the model
    model = CNN(displacement_training_data.shape[1:]).build_model()

    # Compile the model
    model.compile(optimizer=optimizers.Adamax(learning_rate=0.001), loss='mae', metrics='mse')

    # Print the model summary
    model.summary()

    # Early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(displacement_training_data, force_training_data, epochs=1, batch_size=16, callbacks=[stop_early], validation_data=(displacement_validation_data, force_validation_data))

    # Save the trained model
    model.save(file_directory_to_save_results + '/trainedModel.h5')
    print('Saved trainedModel.h5 to %s.' %file_directory_to_save_results)
    
    # Plot the training results
    plot = plots(history, file_directory_to_save_results)
    loss_plot = plot.loss()
    evaluation_metric_plot = plot.evaluation_metric()


if __name__ == '__main__':

    os.system('clear')
    main()
