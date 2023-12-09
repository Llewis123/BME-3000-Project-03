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
from scipy import signal
from scipy.signal import butter, lfilter, iirnotch
from biosppy.signals.ecg import ecg
from scipy.signal import find_peaks


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
    """


    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    general : TYPE, optional
        DESCRIPTION. The default is True.
    all_filters : TYPE, optional
        DESCRIPTION. The default is False.
    diagnostic : TYPE, optional
        DESCRIPTION. The default is False.
    muscle_noise : TYPE, optional
        DESCRIPTION. The default is False.
    Ambulatory : TYPE, optional
        DESCRIPTION. The default is False.
    freq : TYPE, optional
        DESCRIPTION. The default is False.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """

    def notch_filter(data, Q, sampling_rate, notch_frequency=60):
        # Design parameters for the notch filter
        quality_factor = 30.0  # Quality factor for the notch filter

        # Calculate notch filter coefficients
        notch_frequency_normalized = notch_frequency / (0.5 * sampling_rate)
        b, a = iirnotch(notch_frequency_normalized, quality_factor, sampling_rate)

        # Apply the notch filter to the data
        filtered_data = lfilter(b, a, data)

        return filtered_data

    # scipy.signal.iirnotch(w0, Q, fs=2.0)

    def butterworth_filter(data, sampling_rate, cutoff_frequency, order=4, filter_type='low'):
        # Design parameters for the Butterworth filter
        nyquist_frequency = 0.5 * sampling_rate
        cutoff_frequency_normalized = cutoff_frequency / nyquist_frequency

        # Design the Butterworth filter
        b, a = butter(order, cutoff_frequency_normalized, btype=filter_type)

        # Apply the Butterworth filter to the data
        filtered_data = lfilter(b, a, data)

        return filtered_data

    # TODO: Do some filtering here


def getResponses(data, fs=500):
    """
    This function will take a 1D array (1 X N) shape of floats or integers,
    and can represent
    :param data:
    :param fs:
    :return:
    """
    # Create time array
    t = np.arange(0, len(data) / fs, 1 / fs)

    # Impulse response
    _, h_t = signal.impulse((data, [1]), T=t[-1])

    # Frequency response
    f, H_f = signal.freqresp((data, [1]), worN=fft.fftfreq(len(t), 1 / fs))

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Impulse response plot
    axes[0].plot(t, h_t)
    axes[0].set_title('Impulse Response')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    # Frequency response plot
    axes[1].plot(f, np.abs(H_f))
    axes[1].set_title('Frequency Response')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()


def calculate_HRV(r_peaks, fs):
    hrv_analysis = biosppy.signals.hrv(rpeaks=r_peaks, sampling_rate=fs, show=False)
    return hrv_analysis

def get_HRV_BP(hrv_analysis):
    plt.figure(figsize=(8, 6))
    plt.bar(hrv_analysis['frequency'], hrv_analysis['fft_mag'])
    plt.title('HRV Frequency Band Power')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()

def detect_heartbeats(ecg_data, fs):
    # Create time array
    t = np.arange(0, len(ecg_data) / fs, 1 / fs)

    # Process ECG data using biosppy
    ecg_analysis = ecg(signal=ecg_data, sampling_rate=fs, show=False)

    # Get R-peaks
    r_peaks, = find_peaks(ecg_analysis['filtered'], height=0.6)

    # Plot ECG and R-peaks
    plt.figure(figsize=(12, 8))
    plt.plot(t, ecg_data, label='ECG Signal')
    plt.plot(t[r_peaks], ecg_data[r_peaks], 'ro', label='R-peaks')
    plt.title('ECG Signal with R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Calculate HRV
    hrv_analysis = calculate_HRV(r_peaks, fs)

    # Print HRV parameters
    print("HRV Analysis:")
    print(f"Mean RR: {hrv_analysis['mean_rr']} ms")
    print(f"SDNN: {hrv_analysis['sdnn']} ms")
    print(f"RMSSD: {hrv_analysis['rmssd']} ms")

    # Get and plot HRV frequency band power
    get_HRV_BP(hrv_analysis)
