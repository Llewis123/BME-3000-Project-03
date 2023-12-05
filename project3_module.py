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

FOR CHANGELOG: See README.md file attached to module.

IMPORTANT: 
If you are using numpy, scipy, pandas or matplotlib with your project: you only need to import this module.

"""
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

def load_data(filename_1, filename_2 = "", filename_3 = "", filename_4 = "", plot = True, time=True, freq=False, power=False):
  # we will load the data and then plot it. It will be able to read in files of .csv, .txt, .npz
  # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.

  def load_time(filenames):
    # takes in array of filenames to load
    # loads into array, returns it for plotting
  
    return time_array
  
  def load_freq(filenames):
    # takes in array of filenames to load
    # loads into array, returns for plotting
  
    return freq_array


def filter_data(data_set, filter_type="high", impulse_response="finite", cuttoffs, freq = False, dimension=2):
  # here we will take a data_set, which can be an array of any amount of arrays representing data, in any dimension
  # can filter a max of up to five dimensions (for example: MRI data of multiple patients)
  return filtered_data

