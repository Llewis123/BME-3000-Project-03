"""
ANS Evaluation module
By Bila Bogre & Lincoln Lewis
@Llewis123 (github) & @B11LA (github)

This module provides novel functions for analyzing and filtering nervous system activity
This was originally designed and used to:
1. Filter ECG data
2. Automatically detect heartbeat times
3. Calculate HRV - and various time domain characteristics
4. Find HRV power in LF and HF frequency bands
5. Compare LF/HF ratios during periods of relaxation and stress
Additionally;
- load data of any type,
- plot filtered data, next to raw data, next to its frequency domain counterpart
- plot neat bar graphs
- find the frequency response of time domain data
- find the frequency and impulse respnose of a given filter

This module consists of 11 functions.
The first ,load data, can be used to load data of any units (firstly designed for volts)
of the following file types: .csv, .txt, .npz
If needed, the function can load data that is in the frequency domain, or time domain.
It can also plot it given an optional.

FOR CHANGELOG: See data_info.md file attached to module.

IMPORTANT: 
If you are using numpy, scipy, pandas or matplotlib with your project: you only need to import this module.

"""

import biosppy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fft

arduino_IV_CF = (
        1023 / 5
)  # 1023 is the bit resolution of our arduino, 5 is the voltage our arduino 5V pin was outputting
# divide them by eachother to get a conversion factor that we need to convert the analog signal into our digital one
# (mV)


def load_data(filename_1, filename_2, filename_3, filename_4, fs):
    """
    All data must be the same file type
    This does not load individual columns, that you must do on your own, and return as integer arrays.

    Parameters
    ----------
    filename_1 : str
        path to the first data file.
    filename_2 : str
        path to the second data file.
    filename_3 : str
        path to the third data file.
    filename_4 : str
        path to the fourth data file.

    Returns
    -------
    four arrays of int size n,
      Representing the loaded data for the 5 minutes of the activity
      from the specified files.


    """
    # we will load the data and then plot it. It will be able to read in files of .csv, .txt, .npz
    if filename_1.endswith(".txt"):
        activity_1 = np.loadtxt(filename_1)
        activity_2 = np.loadtxt(filename_2)
        activity_3 = np.loadtxt(filename_3)
        activity_4 = np.loadtxt(filename_4)
    elif filename_1.endswith(".csv"):
        activity_1 = np.loadtxt(filename_1, delimiter=",")
        activity_2 = np.loadtxt(filename_2, delimiter=",")
        activity_3 = np.loadtxt(filename_3, delimiter=",")
        activity_4 = np.loadtxt(filename_4, delimiter=",")
    elif filename_1.endswith(".npz"):
        activity_1 = np.load(filename_1)
        activity_2 = np.load(filename_2)
        activity_3 = np.load(filename_3)
        activity_4 = np.load(filename_4)
    else:
        return print("Your file is not one of the specified file types.")
    # returns changed so they get from 5 seconds in to 300 seconds in
    return (
        activity_1[5 * fs:300 * fs] / arduino_IV_CF,
        activity_2[5 * fs:300 * fs] / arduino_IV_CF,
        activity_3[5 * fs:300 * fs] / arduino_IV_CF,
        activity_4[5 * fs:300 * fs] / arduino_IV_CF,
    )

    # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.


def load_x(
        voltage_data, fs, plot=True, freq=False,
):
    """
    Load voltage data and provide options for plotting in the time or frequency domain.


    Parameters
    ----------
    voltage_data : list of numpy arrays
        List of voltage data arrays, each 1-D arrays of floats
    fs : int
        Sampling frequency of the voltage data.
    plot : bool, optional
        Flag to enable or disable plotting. The default is True.
    freq : bool, optional
        Flag to indicate whether to plot in the frequency domain. The default is False.

    Returns
    -------
    concatenated_data : 1D array of floats size 715996,
        Concatenated voltage data from all input arrays.
    x_axis : 1D array od objects size 4,
        Time or frequency values corresponding to the loaded data.
    activities : 2d array of objects
        an array containing the loaded voltage data array of each activity, being 1-D numpy arrays of floats.

    """
    # sets empty  vectors.
    x_axis = np.empty(len(voltage_data), dtype=object)
    activities = np.empty(len(voltage_data), dtype=object)
    concatenated_data = np.concatenate(voltage_data)

    # for indexing.
    index = 0
    # if frequency, loads in and finds the frequency response right away.
    if freq:
        for voltage_set in voltage_data:
            freq = fft.rfftfreq(len(voltage_set), 1 / fs)
            x_axis[index] = freq
            activities[index] = voltage_set
            index += 1
    else: # otherwise, just time_domain.
        for voltage_set in voltage_data:
            time = np.arange(0, len(voltage_set) * 1 / fs, 1 / fs)
            x_axis[index] = time
            activities[index] = voltage_set
            index += 1

    # plots.
    if plot:
        num_subplots = len(voltage_data)

        # Create a grid of subplots based on the number of data arrays
        fig, axs = plt.subplots(
            num_subplots,
            1,
            figsize=(10, 3 * num_subplots),
            clear=True,
            num="loadx",
            sharex="all",
        )

        # Plot each data array on its own subplot
        for i, data_array in enumerate(voltage_data):
            axs[i].plot(data_array, label=f"Data {i + 1}")
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Voltage")
            axs[i].legend()

        # Adjust layout to prevent subplot overlap
        plt.tight_layout()

        # the indexes of the x_axs array matches the indexes of the initial data
        # E.G. the 0th index of voltage data's time array is associated with the 0th index of the x_axs array.
    return concatenated_data, x_axis, activities


def convert_to_db(fft_result):
    """
    convert_to_db will take an fft result and convert it to db units,
    for the purpose of creating bode plots
    This is done by taking the magnitude of our function to get the amplitude,
    and then squaring each amplitude value and normalizing it to max power, while taking
    the absolute value of it

    :param fft_result: n X 1 array of signed 16 bit floats that represent the result of a fast fourier transform,
    where n represents the number of amplitude's in the array :return: the power of the fft result, in decibels,
    an n X 1 array of signed 16 bit floats, where n represents the number of power values in the array
    """

    # get the max power of the array
    max_power = max(fft_result) ** 2
    # find the power_db by taking the log10 of the squared absolute value of the fft_result, divided by our max power
    # to normalize
    power_db = 10 * np.log10(
        np.square(abs(fft_result)) / max_power
    )  # then multiply by 10
    # return
    return power_db


def filter_data(data_set, fs, numtaps, fc, btype="bandpass", window="hann"):
    """
    Applies a filter to the given data set, using a FIR firwin type filter courtesy of scipy.
    This was used to filter ecg data filter, thus used with bandpass and hann window as the defaults.

    Parameters
    ----------
    data_set : 3D-array of arrays, first index represents the first activity, can also be 1-D of floats
        List of data arrays to be filtered.
    fs : float
        Sampling frequency of the data.
    numptaps: int
        Order of the filter, odd number represents an odd numebr of poles,
        even number represents an even number of poles
    Returns
    -------
    filtered_data_set : list of arrays
        List of filtered data arrays, all of the same type that was inputted.
    """

    def filter(data, numtaps, fc, fs, window, btype):
        """
        Apply a FIR filter to the input data, based on given inputs
        It filters it in time_domain, meaning your input must be in time domain

        Parameters
        ----------
        data : 1-D array of floats
            Input data array.
        numtaps : int
            Number of taps (filter order).
        fc : list or float
            Cutoff frequency or frequencies.
        fs : float
            Sampling frequency of the data.
        window : str, optional
            Type of window to use. Available windows can be found on scipy.signal.firwin doc page
             Default is 'hann'.

        Returns
        -------
        filtered_data : array of floats
            FIR-filtered data.
        h_t : array of floats
            Numerial coefficients of the filter, can be used to plot the impulse response
        """
        h_t = signal.firwin(numtaps, fc, window=window, fs=fs, pass_zero=btype)
        filtered = np.convolve(data, h_t, mode="same")
        return filtered, h_t

    filtered_data_set = np.empty(len(data_set), dtype=object)
    # for each data set, use the in-line helper function to filter the data.
    for i, data_array in enumerate(data_set):
        filtered_data_set[i], h_t = filter(data_array, numtaps, fc, fs, window, btype)
    return filtered_data_set, h_t


def plot_domains(data, fs):
    """
   This function takes in a data array and its sampling frequency and plot the
   different domain (frequency, time).

    Parameters
    ----------
    data : 1-D array of floats
        1-D Input data array.
    fs : float
        Sampling frequency of the data.

    Returns
    -------
    None.
    """
    # Calculate time array
    time = np.arange(0, len(data) / fs, 1 / fs)

    # Calculate frequency array
    freq = fft.rfftfreq(len(data), 1 / fs)

    # Calculate Fourier transform of the data
    data_fft = fft.rfft(data)

    # Plot in the time domain
    plt.figure("domains", figsize=(12, 6), clear=True)
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.title("Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot in the frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(freq, convert_to_db(data_fft))
    plt.title("Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.xscale("log")

    plt.tight_layout()


def get_frequency_response(time_domain, fs, dB=True):
    '''

    :param time_domain: a 1-D array of floats representing a given data array (used for ECG data)
    :param fs: float representing the sampling frequency of the given filter
    :param dB: optional, if True, will convert data to decibels, default True.
    :return: a 1-D array of floats, frequencies, representing the frequencies accosiated with the signal
            a 1-D array of floats, frrequency_domain, representing either the power in magnitude or the power in decibels
    '''
    # Calculate frequency array
    frequencies = fft.rfftfreq(len(time_domain), 1 / fs)

    # Calculate Fourier transform of the data
    if dB:  # if we want it in decibels.
        frequency_domain = convert_to_db(fft.rfft(time_domain))
    else:  # otherwise, take the magnitude and get the power.
        frequency_domain = np.abs(np.square(fft.rfft(time_domain)))
    # return both.
    return frequencies, frequency_domain


def plot_filter_response(h_t, fs):
    """
    Plot impulse and frequency response of the filter.

    Parameters
    ----------
    h_t : array of floats representing
        the coefficients of the given filter
    fs: float
        the sampling frequency of the signal, in Hz

    Returns
    -------
    None.
    """

    # create our special time array that centers it so that it goes from -0.5 to 0.5
    t_filter = np.arange((-1 * len(h_t)) / (2 * fs), len(h_t) / (2 * fs), 1 / fs)
    # create filter frequecny
    f_filter = fft.rfftfreq(len(h_t), 1 / fs)
    H_f = fft.rfft(h_t)
    plt.figure("Filters", clear=True)
    # create subplot
    plt.subplot(1, 2, 1)
    # Plot impulse response of our filter
    plt.plot(t_filter, h_t)
    # annotate data
    plt.xlabel("Time (s)")
    plt.ylabel("Impulse response")
    plt.tight_layout()
    plt.grid()

    # create subplot
    plt.subplot(1, 2, 2)
    # Plot frequency response of our filter
    plt.plot(f_filter, convert_to_db(H_f))
    # annotate data
    plt.xlabel("Frequency (hz)")
    plt.ylabel("Power (dB)")
    plt.grid()
    plt.xscale("log")
    plt.tight_layout()


def detect_heartbeats(ecg_data, fs, plot=False):
    """
     Detect and visualize R-peaks in ECG data, and perform HRV analysis.
     This module uses biosppy's  ecg.ecg function (which can be compared with above functions)
     Biosppy is a toolbox for biosignal processing written in Python.
     Its documentation can be found here: https://biosppy.readthedocs.io/en/stable/
     It will return a dictionary of teh given input data with important time-domain results.
     The hrv and ibi were generated based on the biosppys function results as well.

    Parameters
    ----------
    ecg_data : 1-D Array of floats
        the input of the ECG data, in mV.
    fs : float
        Sampling frequency of the ECG data.
    plot : bool, optional
        Flag to enable or disable plotting. Default is False.

    Returns
    -------
    ts : 1-D array of floats
        reference array of time calculated by biosppy based on the sampling frequency,
         same as the time arrays from the other methods (s).
    filtered : 1-D array of floats
        Filtered ECG signal, measured in mV.
    rpeaks : 1-D array representing
        the index location of the R-peaks.
    templates_ts : 1-D array of floats
        Time axis reference for templates (s).
    templates : array
        Templates of heartbeats.
    heart_rate_ts : array
        Time axis reference for heart rate (s).
    heart_rate : float
        Instantaneous heart rate (bpm).
    hrv : int
        Heart Rate Variability response from the ECG analysis.

    """

    # Create our empty dictionary
    ecg_analysis = {}

    # Process ECG data using biosppy
    (
        ts,
        filtered,
        rpeaks,
        templates_ts,
        templates,
        heart_rate_ts,
        heart_rate,
    ) = biosppy.signals.ecg.ecg(signal=ecg_data, sampling_rate=fs, show=plot)
    hrv = np.std(np.diff(ts[rpeaks]))
    ibi = np.diff(ts[rpeaks])
    # add the values to our dictionary
    ecg_analysis["ts"] = ts
    ecg_analysis["filtered"] = filtered
    ecg_analysis["rpeaks"] = rpeaks
    ecg_analysis["templates_ts"] = templates_ts
    ecg_analysis["templates"] = templates
    ecg_analysis["heart_rate_ts"] = heart_rate_ts
    ecg_analysis["heart_rate"] = heart_rate
    ecg_analysis["hrv"] = hrv
    ecg_analysis["ibi"] = ibi

    # Calculate HRV
    return ecg_analysis


def plot_bar(values, categories, title, xlabel="Categories", ylabel="Values"):
    """
    Create a bar plot of a given values and categories array,
    !NOTE: They need to be the same length in order to correctly label each bar plot
    using matplotlibs plt.bar functions
    Parameters
    ----------
    values : floats
        Numeric HRV values to be plotted.
    categories : str
        labels for the activities on the x-axis.
    title : str
        Title of the plot.
    xlabel : str, Opt
        Label for the x-axis. The default is "Categories".
    ylabel : str, Opt
        Label for the y-axis. The default is "Values".

    Returns
    -------
    None.
    """
    # Plotting
    fig, ax = plt.subplots(num="bargraph", figsize=(8, 6), clear=True)

    # Creating a bar plot
    bars = plt.bar(
        categories,
        values,
        color=["skyblue", "grey", "red", "purple"],
        edgecolor="black",
        linewidth=1.2,
    )
    # Adding labels and title
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)

    # Adding data labels on each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02 * max(values),  # Adjusted position for better readability
            f"{yval:.3f}",  # Formatting to two decimal places
            ha="center",
            va="bottom",
        )

    # Adding gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjusting layout
    plt.tight_layout()


# %% part 5
def plot_frequency_bands(
        freq, fft_power, low_fc_range, high_fc_range, title='', ylabel='A.U.'
):
    '''
    plot_frequency_bands plots two different frequency bands of ecg data based on given inputs.
    Will fill with orange and blue colors
    This was used to plot the power of frequency content of an ecg signal, and find the ratio of the
    lower frequency band to the higher frequency band

    :param freq: 1-D array of floats that represent frequency range of a given input
    :param fft_power: 1-D array of floats that represent the fft result of a given input
    :param low_fc_range: 1-D two index list, tupple or array of floats representing teh desires  lower cutoff frequencyies for plotting
    :param high_fc_range:1-D two index list, tupple or array of floats representing teh desires  higher cutoff frequencyies for plotting
    :param title: OPTIONAL string, default is an empty string, represents title of the plots
    :param ylabel: OPTIONAL string, default is 'A.U.'. Represents y label of the pah pah
    :return:
    '''
    # y limits for purpose
    lower = -500
    upper = 8000

    # create boolean masks for the frequency bands
    is_low_fc = (freq >= low_fc_range[0]) & (freq <= low_fc_range[1])
    is_high_fc = (freq >= high_fc_range[0]) & (freq <= high_fc_range[1])

    # use boolean mask to isolate frequency bands
    # low
    low_fc = freq[is_low_fc]
    low_fc_fft = fft_power[is_low_fc]

    # high
    high_fc = freq[is_high_fc]
    high_fc_fft = fft_power[is_high_fc]

    # plot the fft power of the signal
    plt.plot(freq, fft_power, c="gray", zorder=0)

    # plot the frequency bands
    plt.fill_between(low_fc, np.abs(low_fc_fft), label="low frequency band")
    plt.fill_between(high_fc, np.abs(high_fc_fft), label="high frequency band")

    # annotate and format plot
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("frequency (Hz)")
    plt.xlim(0, high_fc_range[1])
    plt.ylim(lower, upper)
    plt.legend()
    plt.grid()

    # compute the mean powers of the frequency bands
    mean_low_fc = np.mean(np.abs(low_fc_fft))
    mean_high_fc = np.mean(np.abs(high_fc_fft))

    # compute and return the low freq/ high freq ratio
    ratio = mean_low_fc / mean_high_fc

    return ratio
