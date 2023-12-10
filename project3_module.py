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
from scipy import signal
from scipy.fft import fft
from scipy.signal import butter, lfilter, iirnotch

arduino_IV_CF = 204.6


def load_data(filename_1, filename_2, filename_3, filename_4):
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
      Representing the loaded data for the 5 minutes of the acyivity 
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

    return (
        activity_1 / arduino_IV_CF,
        activity_2 / arduino_IV_CF,
        activity_3 / arduino_IV_CF,
        activity_4 / arduino_IV_CF,
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
            num_subplots, 1, figsize=(10, 3 * num_subplots), clear=True
        )

        # Plot each data array on its own subplot
        for i, data_array in enumerate(voltage_data):
            axs[i].plot(data_array, label=f"Data {i + 1}")
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Voltage")
            axs[i].legend()

        # Adjust layout to prevent subplot overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
        # the indexes of the x_axs array matches the indexes of the initial data
        # E.G. the 0th index of voltage data's time array is associated with the 0th index of the x_axs array.
    return concatenated_data, x_axis, activities


def filter_data(
    data_set,
    fs,
    general=True,
    all_filters=False,
    diagnostic=False,
    muscle_noise=False,
    Ambulatory=False,
    freq=False,
    plot=True,
):
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

    def notch_filter(data, Q, sampling_rate, notch_frequency=60, coe=False):
        # Design parameters for the notch filter
        quality_factor = 30.0  # Quality factor for the notch filter

        # Calculate notch filter coefficients
        notch_frequency_normalized = notch_frequency / (0.5 * sampling_rate)
        b, a = iirnotch(notch_frequency_normalized, quality_factor, sampling_rate)
        if coe:
            return b, a
        # Apply the notch filter to the data
        filtered_data = lfilter(b, a, data)

        return filtered_data

    # scipy.signal.iirnotch(w0, Q, fs=2.0)

    def butterworth_filter(
        data, sampling_rate, cutoff_frequency, order=4, filter_type="low", coe=False
    ):
        # Design parameters for the Butterworth filter
        nyquist_frequency = 0.5 * sampling_rate
        cutoff_frequency_normalized = cutoff_frequency / nyquist_frequency

        # Design the Butterworth filter
        b, a = butter(order, cutoff_frequency_normalized, btype=filter_type)
        if coe:
            return b, a
        # Apply the Butterworth filter to the data
        filtered_data = lfilter(b, a, data)

        return filtered_data

    # TODO: Do some filtering here

    if general:

        filtered_data_set = np.empty(len(data_set), dtype=object)
        for i, data_array in enumerate(data_set):
            filtered_data_set[i] = butterworth_filter(
                butterworth_filter(data_array, fs, 150), fs, 0.05, filter_type="high"
            )
        b, a = butterworth_filter(data_array[0], fs, 150, coe=True)
        b2, a2 = butterworth_filter(
            data_array[0], fs, 0.05, coe=True, filter_type="high"
        )

        return filtered_data_set, b, a, b2, a2

def plot_domains(data, fs):






def plot_filter_response(b, a, b2=None, a2=None):

    # check to see if this is a cascaded filter (only two allowed)
    if len(b2) != 0:
        # Compute the impulse response of the cascaded system
        b_cascaded = signal.convolve(b, b2)
        a_cascaded = signal.convolve(a, a2)
        _, h_t_cascaded = signal.impulse((b_cascaded, a_cascaded))

        # Compute the frequency response of the cascaded system
        f_cascaded, H_f_cascaded = signal.freqz(b_cascaded, a_cascaded, worN=8000)

        # Plot the impulse response of the cascaded system
        plt.subplot(2, 1, 1)
        plt.plot(h_t_cascaded)
        plt.title("Impulse Response of Cascaded System")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(f_cascaded, np.abs(H_f_cascaded))
        plt.title("Frequency Response of Cascaded System")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()
    else:
        # Compute the impulse response
        _, h_t = signal.impulse((b, a))

        # Compute the time domain response for a unit step input
        t, y_t = signal.step((b, a))

        # Compute the frequency response
        w, H = signal.freqresp((b, a))

        # Plot the impulse response
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(h_t)
        plt.title("Impulse Response")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # Plot the time domain response
        plt.subplot(3, 1, 2)
        plt.plot(t, y_t)
        plt.title("Time Domain Response (Step Input)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # Plot the frequency response
        plt.subplot(3, 1, 3)
        plt.plot(w, np.abs(H))
        plt.title("Frequency Response")
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()


def get_HRV_BP(hrv_analysis):
    plt.figure(figsize=(8, 6))
    plt.bar(hrv_analysis["frequency"], hrv_analysis["fft_mag"])
    plt.title("HRV Frequency Band Power")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.show()


def detect_heartbeats(ecg_data, fs, plot=False):
    # Create time array
    show = False
    if plot:
        show = True
    # Process ECG data using biosppy
    (
        ts,
        filtered,
        rpeaks,
        templates_ts,
        templates,
        heart_rate_ts,
        heart_rate,
    ) = biosppy.signals.ecg.ecg(signal=ecg_data, sampling_rate=fs, show=show)
    hrv = np.std(np.diff(ts[rpeaks]))
    # Calculate HRV
    return (
        ts,
        filtered,
        rpeaks,
        templates_ts,
        templates,
        heart_rate_ts,
        heart_rate,
        hrv
    )
