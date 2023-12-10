import numpy as np
from matplotlib import pyplot as plt

import project3_module as p3m

#%%
fs = 500

activity_1, activity_2, activity_3, activity_4 = p3m.load_data('data_example/sitting_at_rest 3.txt',
                                                               'data_example/relaxing_activity 3.txt',
                                                               'data_example/mentally_stressful 3.txt',
                                                               'data_example/physically_stressful 3.txt')
#%%
concatenated_data, x_axis, activities = p3m.load_x([activity_1, activity_2, activity_3, activity_4], fs = fs)
plt.figure(1, clear=True)
concatenated_time = np.arange(0, len(concatenated_data) * 1/fs, 1/fs)

plt.plot(concatenated_time, concatenated_data)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Concatenated Activities')
plt.grid()
plt.figure(3, clear=True)
plt.xlabel('Time (s)')
plt.ylabel('My filter')
plt.title('Activity 01')
plt.grid()
filtered = p3m.filter_data(activities, 500)
time = np.arange(0, len(filtered[0])* 1/fs, 1/fs)
plt.plot(time, filtered[0])
plt.show()

p3m.get_responses(filtered[0],500)

#%%
ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = p3m.detect_heartbeats(activity_1, fs)
plt.figure(2, clear=True)
plt.plot(ts,filtered)
plt.show()