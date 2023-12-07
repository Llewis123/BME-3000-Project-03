"""
ANS Evaluation module
By Bila Bogre & Lincoln Lewis
@Llewis123 (github) & @B11LA (github)

This module provides novel functions for analyzing and filtering nervous system activity
This was originally designed and used to:
1. Filter ECG data
2. Automatically detect heartbeat times
3. Calculate HRV
4. Find HRV power in LF and HF frequency bands
5. Compare LF/HF ratios during periods of relaxation and stress

This module consists of x functions.
The first ,load data, can be used to load data of any units (firstly designed for volts)
of the following file types: .csv, .txt, .npz
If needed, the function can load data that is in the frequency domain, or time domain.
It can also plot it given an optional.

FOR CHANGELOG: See data_info.md file attached to module.

IMPORTANT: 
If you are using numpy, scipy, pandas or matplotlib with your project: you only need to import this module.

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import fft


def load_data(filename_1, filename_2, filename_3, filename_4):
    """
    All data must be the same file type
    This does not load individual columns, that you must do on your own.

    :param filename_1:
    :param filename_2:
    :param filename_3:
    :param filename_4:
    :param plot:
    :param time:
    :param freq:
    :param power:
    :return:
    """
    # we will load the data and then plot it. It will be able to read in files of .csv, .txt, .npz
    if filename_1.endswith(".txt"):
        activity_1 = np.loadtxt(filename_1)
        activity_2 = np.loadtxt(filename_2)
        activity_3 = np.loadtxt(filename_3)
        activity_4 = np.loadtxt(filename_4)
    elif filename_1.endswith(".csv"):
        activity_1 = np.loadtxt(filename_1, delimiter=',')
        activity_2 = np.loadtxt(filename_2, delimiter=',')
        activity_3 = np.loadtxt(filename_3, delimiter=',')
        activity_4 = np.loadtxt(filename_4, delimiter=',')
    elif filename_1.endswith(".npz"):
        activity_1 = np.load(filename_1)
        activity_2 = np.load(filename_2)
        activity_3 = np.load(filename_3)
        activity_4 = np.load(filename_4)
    else:
        return print("Your file is not one of the specified file types.")

    return activity_1, activity_2, activity_3, activity_4

    # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.


def load_x(voltage_data, fs, plot=True,
           freq=False, power=False, ):
    # takes in array of filenames to load

    x_axis = np.empty(len(voltage_data), dtype=object)
    activities = np.empty(len(voltage_data), dtype=object)
    concatenated_data = np.concatenate(voltage_data)

    index = 0
    if freq:
        for voltage_set in voltage_data:
            freq = fft.rfftfreq(len(voltage_set), 1 / fs)
            x_axis[index] = freq
            activities[index] = voltage_set
            index += 1
    else:
        for voltage_set in voltage_data:
            time = np.arange(0, len(voltage_set) * 1 / fs, 1 / fs)
            x_axis[index] = time
            activities[index] = voltage_set
            index += 1
    # loads into array, returns for plotting
    if plot:
        num_subplots = len(voltage_data)

        # Create a grid of subplots based on the number of data arrays
        fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots), clear=True)

        # Plot each data array on its own subplot
        for i, data_array in enumerate(voltage_data):
            axs[i].plot(data_array, label=f'Data {i + 1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Voltage')
            axs[i].legend()

        # Adjust layout to prevent subplot overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
        # the indexes of the x_axs array matches the indexes of the initial data
        # E.G. the 0th index of voltage data's time array is associated with the 0th index of the x_axs array.
    return concatenated_data, x_axis, activities


def filter_data(data_set, general=True, all_filters=False, diagnostic=False, muscle_noise=False, Ambulatory=False,
                freq=False, plot=True):

    # here we will take a data_set, which can be an array of any amount of arrays representing data
    # filter it and return the filtered data
    return None

# def detect_heartbeats(ecg_data_time, freq_data_time=None plot = True
#
# ):
# here you need filtered ecg_data
# this function can plot the data.
# you can input frequency data if you want it too

# returns array of heartbeat_times#
# return heartbeat_times


# def calculate_HRV(ecg_data, plot=False):
#     return HRV_array, HRV


# def get_HRV_BP(HRV_array, db=False, bar_plot=True):
