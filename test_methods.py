# system imports
import os
import sys

# data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# signal processing
from scipy import signal
from scipy.ndimage import label
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import project3_script as p3m
# misc
import warnings

# style settings
sns.set(style='whitegrid', rc={'axes.facecolor': '#EFF2F7'})

# sample frequency for ECG sensor
settings = {}
settings['fs'] = 500

activity_1, activity_2, activity_3, activity_4 = p3m.load_data('data_example/Arai_sit_relax.txt',
                                                               'data_example/Arai_sit_meditation.txt',
                                                               'data_example/Arai_sit_puzzles.txt',
                                                               'data_example/Arai_bounce_yogaball.txt')

plt.figure(figsize=(20, 7))
start = 0
stop = 5000
duration = (stop-start) / settings['fs']
plt.title("ECG signal, slice of %.1f seconds" % duration)
plt.plot(activity_1[start:stop].index, activity_1[start:stop].heartrate, color="#51A6D8", linewidth=1)
plt.xlabel("Time (ms)", fontsize=16)
plt.ylabel("Amplitude (arbitrary unit)")
plt.show()

def detect_peaks(ecg_signal, threshold=0.3, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # return peaks (values in ms) using threshold
    return ecg_signal[similarity > threshold].index, similarity

def get_plot_ranges(start=10, end=20, n=5):
    '''
    Make an iterator that divides into n or n+1 ranges.
    - if end-start is divisible by steps, return n ranges
    - if end-start is not divisible by steps, return n+1 ranges, where the last range is smaller and ends at n

    # Example:
    >> list(get_plot_ranges())
    >> [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]

    '''
    distance = end - start
    for i in np.arange(start, end, np.floor(distance/n)):
        yield (int(i), int(np.minimum(end, np.floor(distance/n) + i)))

sampfrom = 60000
sampto = 70000
nr_plots = 1

for start, stop in get_plot_ranges(sampfrom, sampto, nr_plots):
    # get slice data of ECG data
    cond_slice = (activity_1.index >= start) & (activity_1.index < stop)
    ecg_slice = activity_1.heartrate[cond_slice]

    # detect peaks
    peaks, similarity = detect_peaks(ecg_slice, threshold=0.3)

    # plot similarity
    plt.figure(figsize=(20, 15))

    plt.subplot(211)
    plt.title("ECG signal with found peaks")
    plt.plot(ecg_slice.index, ecg_slice, label="ECG", color="#51A6D8", linewidth=1)
    plt.plot(
        peaks,
        np.repeat(600, peaks.shape[0]),
        label="peaks",
        color="orange",
        marker="o",
        linestyle="None",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Amplitude (arbitrary unit)")

    plt.subplot(212)
    plt.title("Similarity with QRS template")
    plt.plot(
        ecg_slice.index,
        similarity,
        label="Similarity with QRS filter",
        color="olive",
        linewidth=1,
    )
    plt.legend(loc="upper right")
    plt.xlabel("Time (milliseconds)")
    plt.ylabel("Similarity (normalized)")


    sampfrom = 60000
    sampto = 70000
    nr_plots = 1
    
    for start, stop in get_plot_ranges(sampfrom, sampto, nr_plots):
        # get slice data of ECG data
        cond_slice = (activity_1.index >= start) & (activity_1.index < stop)
        ecg_slice = activity_1.heartrate[cond_slice]
    
        # detect peaks
        peaks, similarity = detect_peaks(ecg_slice, threshold=0.3)
    
        # plot similarity
        plt.figure(figsize=(20, 15))
    
        plt.subplot(211)
        plt.title("ECG signal with found peaks")
        plt.plot(ecg_slice.index, ecg_slice, label="ECG", color="#51A6D8", linewidth=1)
        plt.plot(
            peaks,
            np.repeat(600, peaks.shape[0]),
            label="peaks",
            color="orange",
            marker="o",
            linestyle="None",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time (milliseconds)")
        plt.ylabel("Amplitude (arbitrary unit)")
    
        plt.subplot(212)
        plt.title("Similarity with QRS template")
        plt.plot(
            ecg_slice.index,
            similarity,
            label="Similarity with QRS filter",
            color="olive",
            linewidth=1,
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time (milliseconds)")
        plt.ylabel("Similarity (normalized)")


        # detect peaks
        peaks, similarity = detect_peaks(activity_1.heartrate, threshold=0.3)
        
        # group peaks
        grouped_peaks = group_peaks(peaks)
        
        # plot peaks
        plt.figure(figsize=(20, 7))
        plt.title("Group similar peaks together")
        plt.plot(activity_1.index, activity_1.heartrate, label="ECG", color="#51A6D8", linewidth=2)
        plt.plot(
            peaks,
            np.repeat(600, peaks.shape[0]),
            label="samples above threshold (found peaks)",
            color="orange",
            marker="o",
            linestyle="None",
        )
        plt.plot(
            grouped_peaks,
            np.repeat(620, grouped_peaks.shape[0]),
            label="median of found peaks",
            color="k",
            marker="v",
            linestyle="None",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (arbitrary unit)")
        plt.gca().set_xlim(0, 200)
        plt.show()