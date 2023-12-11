import numpy as np
from matplotlib import pyplot as plt

import project3_module as p3m

#%%
fs = 500

sit_at_rest, relaxing_activity, mentally_stressful, physically_stressful = p3m.load_data(
    "data_example/sitting_at_rest 3.txt",
    "data_example/relaxing_activity 3.txt",
    "data_example/mentally_stressful 3.txt",
    "data_example/physically_stressful 3.txt",
    500,
)
#%%
concatenated_data, x_axis, activities = p3m.load_x(
    [sit_at_rest, relaxing_activity, mentally_stressful, physically_stressful], fs=fs, plot=False
)
concatenated_time = np.arange(0, len(concatenated_data) * 1 / fs, 1 / fs)

num_subplots = len(activities)

plt.figure(clear=True)
# Create a grid of subplots based on the number of data arrays
fig, axs = plt.subplots(num_subplots, 2, figsize=(10, 3 * num_subplots), clear=True, sharex='all')
time = np.arange(0, len(sit_at_rest) / fs, 1 / fs)
filtered= p3m.filter_data(activities, 500)
# Plot each data array on its own subplot
for index, data_array in enumerate(activities):
    axs[index, 0].plot(time, data_array, label=f"Data {index + 1}")
    axs[index, 0].set_xlabel("Time")
    axs[index, 0].set_ylabel("Voltage")
    axs[index, 0].legend()
    # Plot filtered data in the second column
    axs[index, 1].plot(time, filtered[index], label=f"Filtered Data {index + 1}")
    axs[index, 1].set_xlabel("Time")
    axs[index, 1].set_ylabel("Voltage")
    axs[index, 1].legend()

# Adjust layout to prevent subplot overlap
plt.tight_layout()

plt.figure(clear=True)

# Show the plots
plt.plot(concatenated_time, concatenated_data)
plt.xlabel("Time (s)")
plt.ylabel("Raw Data")
plt.title("Concatenated Activities")
plt.grid()

#%%
''' 
Activity 1
'''
ecg_analysis_A1= p3m.detect_heartbeats(sit_at_rest, fs, plot=False)

#%%
'''
Activity 2
'''
ecg_analysis_A2 = p3m.detect_heartbeats(relaxing_activity, fs, plot=False)

#%%
'''
Activity 3
'''
ecg_analysis_A3 = p3m.detect_heartbeats(mentally_stressful, fs, plot=False)

#%%
'''
Activity 4
'''
ecg_analysis_A4 = p3m.detect_heartbeats(physically_stressful, fs, plot=False)


plt.show()
