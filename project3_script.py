"""
ANS Evaluation module
By Bila Bogre & Lincoln Lewis
@Llewis123 (github) & @B11LA (github)
This script runs the HRV analysis on data from various activities. It makes use of a custom module called "project3_module" for data 
loading, filtering, heartbeat detection, and HRV calculation. For each activity, the script includes visualizations of filtered signals, 
R-peaks detection, frequency domain analysis, and HRV bar plots.

Courtesy biosppy and their beautiful ecg package that was used in this project, at https://biosppy.readthedocs.io/en/stable/
we used our own filter as well in this, and compared the results as we were curious.
Ill have to say that our filter is kinda better at lower noise but not wen their is a lot of nosie present.
"""
import numpy as np
from matplotlib import pyplot as plt

import project3_module as p3m

#%%
""" 
PART 1
"""
fs = 500

(
    sit_at_rest,
    relaxing_activity,
    mentally_stressful,
    physically_stressful,
) = p3m.load_data(
    "data_example/sitting_at_rest 3.txt",
    "data_example/relaxing_activity 3.txt",
    "data_example/mentally_stressful 3.txt",
    "data_example/physically_stressful 3.txt",
    500,
)
concatenated_data, x_axis, activities = p3m.load_x(
    [sit_at_rest, relaxing_activity, mentally_stressful, physically_stressful],
    fs=fs,
    plot=False,
)
concatenated_time = np.arange(0, len(concatenated_data) * 1 / fs, 1 / fs)
#%%
""" 
PART 2
We will plot the filtered and raw signal in one figure here.
"""

num_subplots = len(activities)
# Create a grid of subplots based on the number of data arrays
fig, axs = plt.subplots(
    num_subplots,
    3,
    num="Filtered vs normal",
    figsize=(4 * num_subplots, 8),
    clear=True,
    sharex="col",
)
# set time arrays, get filtered dataset and h_t
# because h_t will be the same for each, we don't really care here, in terms of the final return value of h_t
time = np.arange(0, len(sit_at_rest) / fs, 1 / fs)
filtered, h_t = p3m.filter_data(activities, fs, 100, [0.04, 0.4, 0.7, 27])
# Plot each data array on its own subplot
for index, data_array in enumerate(activities):
    axs[index, 0].plot(
        time[45 * fs : 50 * fs],
        data_array[45 * fs : 50 * fs],
        label=f"Data {index + 1}",
    )
    axs[index, 0].plot(
        time[45 * fs : 50 * fs],
        filtered[index][45 * fs : 50 * fs],
        label=f"Filtered Data {index + 1}",
        alpha=0.7,
        color="orange",
    )
    axs[index, 0].set_xlabel("Time")
    axs[index, 0].set_ylabel("Voltage")
    axs[index, 0].legend()
    # Plot filtered data in the second column
    axs[index, 1].plot(
        time[45 * fs : 50 * fs],
        filtered[index][45 * fs : 50 * fs],
        label=f"Filtered Data {index + 1} (our filter)",
    )
    axs[index, 1].set_xlabel("Time")
    axs[index, 1].set_ylabel("Voltage")
    axs[index, 1].legend()
    freq, freq_response = p3m.get_frequency_response(data_array, fs)
    axs[index, 2].plot(
        freq, freq_response, label=f"Frequency domain {index + 1} (no filter)"
    )
    axs[index, 2].set_xlabel("Frequency")
    axs[index, 2].set_xscale("log")
    axs[index, 2].set_ylabel("Power (dB)")
    axs[index, 2].legend()

# Adjust layout to prevent subplot overlap
plt.tight_layout()
plt.figure("Concate", clear=True)

# Show the plots
plt.plot(concatenated_time, concatenated_data)
plt.xlabel("Time (s)")
plt.ylabel("Raw Data")
plt.title("Concatenated Activities")
plt.grid()
# plot our filters response
p3m.plot_filter_response(h_t, fs)

"""
PART 3, 4 and 5
We contained these parts into this section,
they are ordered by the activity with every part done for each activity
for reading purposes

"""
#%%
""" 
Sit at rest
"""
# get the first activities ecg_analysis
ecg_analysis_A1 = p3m.detect_heartbeats(sit_at_rest, fs)
# assign
A1_rpeaks = ecg_analysis_A1["rpeaks"]
A1_filtered = ecg_analysis_A1["filtered"]
# plot the results, and annotate.
plt.figure("A1", figsize=(15, 8), clear=True)
plt.plot(time, A1_filtered, label="Filtered (biosppy) sit at rest signal")
plt.plot(time[A1_rpeaks], A1_filtered[A1_rpeaks], "x", label="R-peaks")
plt.title("Activity one signal (filtered) with R-Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.legend()

#variable for what we want our new dt to be
new_dt = 0.1
# new time course
time_course = np.arange(0, len(time) * new_dt, new_dt)
# get the inter beat intervals
ibi = ecg_analysis_A1["ibi"]
# interpolate this with the time_course, to get a conistent sampling frequency, in order to find fft
ibi_interpolated = np.interp(time_course, time[A1_rpeaks][1:],
                             ibi)
# get the frequency respeonse of this activity
freq_hrv, hrv_mag = p3m.get_frequency_response(ibi_interpolated, fs, dB=False)
# plot and annotate using project 3 module
plt.figure("A1HPV", figsize=(10, 8), clear=True)
ratio_A1 = p3m.plot_frequency_bands(
    freq_hrv * new_dt,
    hrv_mag,
    [0.035, 0.15],
    [0.15, 0.4],
    title="Sitting at rest power vs freq bands",
    units="mV^2",
)
# REST OF COMMENTS WOULD BE SAME EXACT FROM HERE ON OUT PLEASEAE AWE AW

#%%
# Plotting each activity with the frequency bands from the module

"""
Relaxing activity
"""
ecg_analysis_A2 = p3m.detect_heartbeats(relaxing_activity, fs)
A2_rpeaks = ecg_analysis_A2["rpeaks"]
A2_filtered = ecg_analysis_A2["filtered"]
plt.figure("A2", figsize=(15, 8), clear=True)
plt.plot(time, A2_filtered, label="Filtered (biosppy) relaxing activity signal")
plt.plot(time[A2_rpeaks], A2_filtered[A2_rpeaks], "x", label="R-peaks")
plt.title("Activity two signal (filtered) with R-Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.legend()

time_course = np.arange(0, len(time) * new_dt, new_dt)
ibi = ecg_analysis_A2["ibi"]
ibi_interpolated = np.interp(time_course, time[A2_rpeaks][1:], ibi)
freq_hrv, hrv_mag = p3m.get_frequency_response(ibi_interpolated, fs, dB=False)

plt.figure("A2HPV", figsize=(10, 8), clear=True)
ratio_A2 = p3m.plot_frequency_bands(
    freq_hrv * new_dt,
    hrv_mag,
    [0.035, 0.15],
    [0.15, 0.4],
    title="Relaxing activity power vs freq bands",
    units="mV^2",
)


#%%
"""
Mentally stressful
"""
ecg_analysis_A3 = p3m.detect_heartbeats(mentally_stressful, fs)
A3_rpeaks = ecg_analysis_A3["rpeaks"]
A3_filtered = ecg_analysis_A3["filtered"]
plt.figure("A3", figsize=(15, 8), clear=True)
plt.plot(time, A3_filtered, label="Filtered (biosppy) mentally stressful signal")
plt.plot(time[A3_rpeaks], A3_filtered[A3_rpeaks], "x", label="R-peaks")
plt.title("Activity three signal (filtered) with R-Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.legend()

time_course = np.arange(0, len(time) * new_dt, new_dt)
ibi = ecg_analysis_A3["ibi"]
ibi_interpolated = np.interp(time_course, time[A3_rpeaks][1:], ibi)
freq_hrv, hrv_mag = p3m.get_frequency_response(ibi_interpolated, fs, dB=False)

plt.figure("A3HPV", figsize=(10, 8), clear=True)
ratio_A3 = p3m.plot_frequency_bands(
    freq_hrv * new_dt,
    hrv_mag,
    [0.035, 0.15],
    [0.15, 0.4],
    title="Mentally stressful activity power vs freq bands",
    units="mV^2",
)


#%%
"""
Physically stressful
"""
ecg_analysis_A4 = p3m.detect_heartbeats(physically_stressful, fs, plot=True)
A4_rpeaks = ecg_analysis_A4["rpeaks"]
A4_filtered = ecg_analysis_A4["filtered"]
plt.figure("A4", figsize=(15, 8), clear=True)
plt.plot(time, A4_filtered, label="Filtered (biosppy) Physically stressful signal")
plt.plot(time[A4_rpeaks], A4_filtered[A4_rpeaks], "x", label="R-peaks")
plt.title("Activity four signal (filtered) with R-Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.legend()

time_course = np.arange(0, len(time) * new_dt, new_dt)
ibi = ecg_analysis_A4["ibi"]
ibi_interpolated = np.interp(time_course, time[A4_rpeaks][1:], ibi)
freq_hrv, hrv_mag = p3m.get_frequency_response(ibi_interpolated, fs, dB=False)

plt.figure("A4HPV", figsize=(10, 8), clear=True)
ratio_A4 = p3m.plot_frequency_bands(
    freq_hrv * new_dt,
    hrv_mag,
    [0.035, 0.15],
    [0.15, 0.4],
    title="physically stressful activity power vs freq bands",
    units="mV^2",
)


#%%
# plot the bar plot for the ratios and HRV values from the module
"""
HRV Bar plot
"""
HRV_values = [
    ecg_analysis_A1["hrv"],
    ecg_analysis_A2["hrv"],
    ecg_analysis_A3["hrv"],
    ecg_analysis_A4["hrv"],
]
categories = [
    "Sitting at rest",
    "Relaxing activity",
    "Mentally Stressful Activity",
    "Physically stressful activity",
]
p3m.plot_bar(HRV_values, categories, "HRV vs activities")

#%%
"""
Ratio bar plots
"""
HRV_values = [ratio_A1, ratio_A2, ratio_A3, ratio_A4]
categories = [
    "Sitting at rest",
    "Relaxing activity",
    "Mentally Stressful Activity",
    "Physically stressful activity",
]
p3m.plot_bar(HRV_values, categories, "HRV LF/HF ratios")

# show all of our plots,
# save figure was done finally and then erased to avoid annoyance when debugging.
plt.show()
