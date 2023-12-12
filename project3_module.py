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

import biosppy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fft

arduino_IV_CF = (
    1023 / 5
)  # 1023 is the bit resolution of our arduino, 5 is the voltage our arduino 5V pin was outputting
# divide them by eachother to get a conversion factor that we need to convert the analog signal into our digital one (mV)


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
        sit_at_rest = np.loadtxt(filename_1)
        relaxing_activity = np.loadtxt(filename_2)
        metally_stressful = np.loadtxt(filename_3)
        physically_stressful = np.loadtxt(filename_4)
    elif filename_1.endswith(".csv"):
        sit_at_rest = np.loadtxt(filename_1, delimiter=",")
        relaxing_activity = np.loadtxt(filename_2, delimiter=",")
        metally_stressful = np.loadtxt(filename_3, delimiter=",")
        physically_stressful = np.loadtxt(filename_4, delimiter=",")
    elif filename_1.endswith(".npz"):
        sit_at_rest = np.load(filename_1)
        relaxing_activity = np.load(filename_2)
        metally_stressful = np.load(filename_3)
        physically_stressful = np.load(filename_4)
    else:
        return print("Your file is not one of the specified file types.")
    # returns changed so they get from 5 seconds in to 300 seconds in
    return (
        sit_at_rest[5*fs:300*fs] / arduino_IV_CF,
        relaxing_activity[5*fs:300*fs] / arduino_IV_CF,
        metally_stressful[5*fs:300*fs] / arduino_IV_CF,
        physically_stressful[5*fs:300*fs] / arduino_IV_CF,
    )

    # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.


def load_x(
    voltage_data, fs, plot=True, freq=False, power=False,
):
    """
    Load voltage data and provide options for plotting in the time or frequency domain.


    Parameters
    ----------
    voltage_data : list of numpy arrays
        List of voltage data arrays.
    fs : int
        Sampling frequency of the voltage data.
    plot : bool, optional
        Flag to enable or disable plotting. The default is True.
    freq : bool, optional
        Flag to indicate whether to plot in the frequency domain. The default is False.
    power : bool, optional
        Placeholder parameter. The default is False.

    Returns
    -------
    concatenated_data : 1D array of floats size 715996,
        Concatenated voltage data from all input arrays.
    x_axis : 1D array od objects size 4,
        Time or frequency values corresponding to the loaded data.
    activities : 2d array of objects
        an array containing the loaded voltage data array of each activity.

    """
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
    Apply signal filters to the input data set and provide optional visualization.

    Parameters
    ----------
    data_set : list of arrays
        List of data arrays to be filtered.
    fs : float
        Sampling frequency of the data.
    general : bool, optional
        Flag to apply a general FIR filter. Default is True.
    all_filters : bool, optional
        Placeholder parameter. Default is False.
    diagnostic : bool, optional
        Placeholder parameter. Default is False.
    muscle_noise : bool, optional
        Placeholder parameter. Default is False.
    Ambulatory : bool, optional
        Placeholder parameter. Default is False.
    freq : bool, optional
        Flag to indicate whether to plot in the frequency domain. Default is False.
    plot : bool, optional
        Flag to enable or disable plotting. Default is True.

    Returns
    -------
    filtered_data_set : list of arrays
        List of filtered data arrays.
    """

    def filter(data, numtaps, fc, fs, window, btype):
        """
        Apply a FIR filter to the input data.

        Parameters
        ----------
        data : array
            Input data array.
        numtaps : int
            Number of taps (filter order).
        fc : list or float
            Cutoff frequency or frequencies.
        fs : float
            Sampling frequency of the data.
        window : str, optional
            Type of window to use. Default is 'hann'.

        Returns
        -------
        filtered_data : array
            FIR-filtered data.
        h_t : array
            Impulse response of the filter.
        """
        h_t = signal.firwin(numtaps, fc, window=window, fs=fs, pass_zero=btype)
        filtered = np.convolve(data, h_t, mode="same")
        return filtered, h_t

    filtered_data_set = np.empty(len(data_set), dtype=object)
    for i, data_array in enumerate(data_set):
        filtered_data_set[i], h_t = filter(data_array, numtaps, fc, fs, window, btype)
    return filtered_data_set, h_t


def plot_domains(data, fs):
    """
    Plot time and frequency domains of the input data.

    Parameters
    ----------
    data : array
        Input data array.
    fs : float
        Sampling frequency of the data.

    Returns
    -------
    None.
    """
    # Calculate time array
    t = np.arange(0, len(data) / fs, 1 / fs)

    # Calculate frequency array
    f = fft.rfftfreq(len(data), 1 / fs)

    # Calculate Fourier transform of the data
    data_fft = fft.rfft(data)

    # Plot in the time domain
    plt.figure("domains", figsize=(12, 6), clear=True)
    plt.subplot(2, 1, 1)
    plt.plot(t, data)
    plt.title("Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot in the frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(f, convert_to_db(data_fft))
    plt.title("Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.xscale("log")

    plt.tight_layout()


def get_frequency_response(time_domain, fs, dB=True):

    # Calculate frequency array
    f = fft.rfftfreq(len(time_domain), 1/fs)

    # Calculate Fourier transform of the data
    if dB:
        data_fft = convert_to_db(fft.rfft(time_domain))
    else:
        data_fft = np.abs(fft.rfft(time_domain))

    return f, data_fft


def plot_filter_response(h_t, fs):
    """
    Plot impulse and frequency response of the filter.

    Parameters
    ----------
    b, a : arrays
        Numerator and denominator coefficients of the filter.
    b2, a2 : arrays, optional
        Second set of coefficients for cascaded filters.

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


    Parameters
    ----------
    ecg_data : Array
        the inpur of the ECG data.
    fs : int
        Sampling frequency of the ECG data.
    plot : bool, optional
        Flag to enable or disable plotting. Default is False.

    Returns
    -------
    ts : array
        an array indicating the signal time (s).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        the location of the R-peaks.
    templates_ts : array
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
    Create a bar plot.

    Parameters
    ----------
    values : floats
        Numeric HRV values to be plotted.
    categories : str
        labels for the activities on the x-axis.
    title : str
        Title of the plot.
    xlabel : str, 
        Label for the x-axis. The default is "Categories".
    ylabel : str, 
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

    # Adding legend if needed
    # plt.legend(['Legend 1', 'Legend 2'], loc='upper right')

    # Adjusting layout
    plt.tight_layout()

#%% part 5
def plot_frequency_bands(freq, fft_power, low_fc_range, high_fc_range, title = None, units = 'A.U.'):
    '''
    This function computes the FFT of the input signal and plots the power in the frequency domain.
    It also isolates a low and high frequency band as specified by the inputs
    and displays those regions on the plot. Then the mean power of the band is calculated
    and the ratio of low frequency power to high frequency power is computed. This ratio
    is used to approximate sympathetic nervous system activity based on the inter-beat interval
    signal from an ECG recording.

    Parameters
    ----------
    signal : 1D array of floats size (n,) where n is the number of samples in the signal
        Signal to compute FFT and isolate frequency bands on.
    fs : integer
        The sampling frequency of the signal.
    low_fc_range : 1D list or array of floats size 2 or shape (2,)
        The bounds of the low frequency band to be extracted.
    high_fc_range : 1D list or array of floats size 2 or shape (2,)
        The bounds of the high frequency band to be extracted.
    title : string, optional
        The title of the plot that will be created of the input signal in the frequency domain.
        The default is None.
    units : string, optional
        The y axis label containg the units of the y axis. The default is 'A.U.'.

    Returns
    -------
    ratio : float
        The low to high frequency ratio of the mean power within the frequency bands.

    '''
    # compute the fft of the signal and the corresponding frequencies for the x axis

    # create boolean masks for the frequency bands
    is_low_fc_mask = (freq >= low_fc_range[0]) & (freq <= low_fc_range[1])
    is_high_fc_mask = (freq >= high_fc_range[0]) & (freq <= high_fc_range[1])

    # use boolean mask to isolate frequency bands
    low_fc_fft = fft_power[is_low_fc_mask]
    low_fc = freq[is_low_fc_mask]
    high_fc_fft = fft_power[is_high_fc_mask]
    high_fc = freq[is_high_fc_mask]


    # plot the fft power of the signal
    plt.plot(freq, fft_power, c = 'gray', zorder = 0 )

    # plot the frequency bands
    plt.fill_between(low_fc, np.abs(low_fc_fft), label = 'low frequecy band')
    plt.fill_between(high_fc, np.abs(high_fc_fft), label = 'high frequency band')

    # annotate and format plot
    plt.title(title)
    plt.ylabel(units)
    plt.xlabel('frequency (Hz)')
    plt.xlim(0,high_fc_range[1])
    plt.legend()
    plt.grid()

    # compute the mean powers of the frequency bands
    mean_low_fc = np.mean(np.abs(low_fc_fft))
    mean_high_fc = np.mean(np.abs(high_fc_fft))

    # compute and return the low freq/ high freq ratio
    ratio = mean_low_fc / mean_high_fc

    return ratio