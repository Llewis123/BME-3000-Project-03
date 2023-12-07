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


def load_data(filename_1, filename_2, filename_3, filename_4, plot=True, time=True, freq=False, power=False,
              ext=".txt"):
    # we will load the data and then plot it. It will be able to read in files of .csv, .txt, .npz
    if ext == ".txt":
        activity_1 = np.loadtxt(filename_1)
        activity_2 = np.loadtxt(filename_2)
        activity_3 = np.loadtxt(filename_3)
        activity_4 = np.loadtxt(filename_4)
    elif ext == ".csv":
        activity_1 = np.loadtxt(filename_1, delimiter=',')
        activity_2 = np.loadtxt(filename_2, delimiter=',')
        activity_3 = np.loadtxt(filename_3, delimiter=',')
        activity_4 = np.loadtxt(filename_4, delimiter=',')
    elif ext == ".npz":
        activity_1 = np.load(filename_1)
        activity_2 = np.load(filename_2)
        activity_3 = np.load(filename_3)
        activity_4 = np.load(filename_4)
    else:
        return print("Your file is not one of the specified file types.")


    return activity_1, activity_2, activity_3, activity_4

    # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.


def load_x(time, filename_1, filename_2, filename_3, filename_4):
    # takes in array of filenames to load
    time = np.arange(0, 5, 1 / 100)
    # np.arange(0, len(load_cell_data) * 1/fs, 1/fs)

    concatenated_data = np.concatenate([activity_1, activity_2, activity_3, activity_4])

    return concatenated_data

    # loads into array, returns for plotting

    return freq_array


def filter_data(data_set, filter_type="high", impulse_response="finite", cuttoffs, freq=False, plot=True):
    # here we will take a data_set, which can be an array of any amount of arrays representing data
    # filter it and return the filtered data
    return filtered_data


def detect_heartbeats(ecg_data_time, freq_data_time=None plot = True

):
# here you need filtered ecg_data
# this function can plot the data.
# you can input frequency data if you want it too

# returns array of heartbeat_times
return heartbeat_times


def calculate_HRV(ecg_data, plot=False):
    return HRV_array, HRV


def get_HRV_BP(HRV_array, db=False, bar_plot=True):
