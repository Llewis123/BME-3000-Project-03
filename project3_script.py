import heartpy as hp
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
)
#%%
concatenated_data, x_axis, activities = p3m.load_x(
    [activity_1, activity_2, activity_3, activity_4], fs=fs, plot=False
)
plt.figure(1, clear=True)
concatenated_time = np.arange(0, len(concatenated_data) * 1 / fs, 1 / fs)

plt.plot(concatenated_time, concatenated_data)
plt.xlabel("Time (s)")
plt.ylabel("Raw Data")
plt.title("Concatenated Activities")
plt.grid()
plt.show()
filtered, b, a, b2, a2 = p3m.filter_data(activities, 500)
p3m.plot_filter_response(b,a,b2,a2)
#%%
(
    ts,
    filtered,
    rpeaks,
    templates_ts,
    templates,
    heart_rate_ts,
    heart_rate,
    hrv
) = p3m.detect_heartbeats(activity_4, fs, plot=True)

print(hrv)