import numpy as np
from matplotlib import pyplot as plt

import project3_module as p3m

#%%
fs = 500

activity_1, activity_2, activity_3, activity_4 = p3m.load_data(
    "data_example/sitting_at_rest 3.txt",
    "data_example/relaxing_activity 3.txt",
    "data_example/mentally_stressful 3.txt",
    "data_example/physically_stressful 3.txt",
    500,
)
#%%
concatenated_data, x_axis, activities = p3m.load_x(
    [activity_1, activity_2, activity_3, activity_4], fs=fs, plot=False
)
concatenated_time = np.arange(0, len(concatenated_data) * 1 / fs, 1 / fs)

num_subplots = len(activities)

# Create a grid of subplots based on the number of data arrays
fig, axs = plt.subplots(num_subplots, 3,num = "Filtered vs normal",  figsize=(4*num_subplots, 8), clear=True, sharex='col')
time = np.arange(0, len(activity_1) / fs, 1 / fs)
filtered, h_t = p3m.filter_data(activities, fs, 100, [0.04, 0.4, 0.7, 27])
# Plot each data array on its own subplot
for index, data_array in enumerate(activities):
    axs[index, 0].plot(time[45*fs:50*fs], data_array[45*fs:50*fs], label=f"Data {index + 1}")
    axs[index, 0].plot(time[45*fs:50*fs], filtered[index][45*fs:50*fs], label=f"Filtered Data {index + 1}", alpha = 0.7, color = 'orange')
    axs[index, 0].set_xlabel("Time")
    axs[index, 0].set_ylabel("Voltage")
    axs[index, 0].legend()
    # Plot filtered data in the second column
    axs[index, 1].plot(time[45*fs:50*fs], filtered[index][45*fs:50*fs], label=f"Filtered Data {index + 1} (our filter)")
    axs[index, 1].set_xlabel("Time")
    axs[index, 1].set_ylabel("Voltage")
    axs[index, 1].legend()
    freq, freq_response = p3m.get_frequency_response(data_array, fs)
    axs[index, 2].plot(freq, freq_response, label=f"Frequency domain {index + 1} (no filter)")
    axs[index, 2].set_xlabel("Frequency")
    axs[index, 2].set_xscale('log')
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

#%%
''' 
Activity 1
'''
ecg_analysis_A1= p3m.detect_heartbeats(activity_1, fs)
A1_rpeaks = ecg_analysis_A1['rpeaks']
A1_filtered = ecg_analysis_A1['filtered']
plt.figure("A1", figsize=(15,8), clear=True)
plt.plot(time, A1_filtered, label='Filtered (biosppy) Activity_1 signal')
plt.plot(time[A1_rpeaks], A1_filtered[A1_rpeaks], 'x', label='R-peaks')
plt.title("Activity one signal (filtered) with R-Peaks")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()

plt.figure("A1HPV", figsize=(10,8), clear=True)
time_course = np.arange(0, len(A1_filtered)/ fs, 0.1)
ibi = ecg_analysis_A1['ibi']
ibi_interpolated = np.interp(time_course, time[A1_rpeaks][1:], ibi)
freq_hrv, hrv_mag = p3m.get_frequency_response(ibi_interpolated, fs, dB=False)
plt.plot(freq_hrv, hrv_mag, label='Filtered (biosppy) Activity_1 signal')


#%%
'''
Activity 2
'''
ecg_analysis_A2 = p3m.detect_heartbeats(activity_2, fs)
A2_rpeaks = ecg_analysis_A2['rpeaks']
A2_filtered = ecg_analysis_A2['filtered']
plt.figure("A2", figsize=(15,8), clear=True)
plt.plot(time, A2_filtered, label='Filtered (biosppy) Activity_2 signal')
plt.plot(time[A2_rpeaks], A2_filtered[A2_rpeaks], 'x', label='R-peaks')
plt.title("Activity two signal (filtered) with R-Peaks")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()

#%%
'''
Activity 3
'''
ecg_analysis_A3 = p3m.detect_heartbeats(activity_3, fs)
A3_rpeaks = ecg_analysis_A3['rpeaks']
A3_filtered = ecg_analysis_A3['filtered']
plt.figure("A3", figsize=(15,8), clear=True)
plt.plot(time, A3_filtered, label='Filtered (biosppy) Activity_3 signal')
plt.plot(time[A3_rpeaks], A3_filtered[A3_rpeaks], 'x', label='R-peaks')
plt.title("Activity three signal (filtered) with R-Peaks")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()

#%%
'''
Activity 4
'''
ecg_analysis_A4 = p3m.detect_heartbeats(activity_4, fs, plot=True)
A4_rpeaks = ecg_analysis_A4['rpeaks']
A4_filtered = ecg_analysis_A4['filtered']
plt.figure("A4", figsize=(15,8), clear=True)
plt.plot(time, A4_filtered, label='Filtered (biosppy) Activity_4 signal')
plt.plot(time[A4_rpeaks], A4_filtered[A4_rpeaks], 'x', label='R-peaks')
plt.title("Activity four signal (filtered) with R-Peaks")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()


#%%
'''
HRV Bar plot
'''
HRV_values = [ecg_analysis_A1['hrv'],
              ecg_analysis_A2['hrv'],
              ecg_analysis_A3['hrv'],
              ecg_analysis_A4['hrv']]
categories = ['activity one',
              'activity two',
              'activity three',
              'activity four']
p3m.plot_bar(HRV_values, categories, 'HRV vs activities')
plt.show()
