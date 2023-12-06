import project3_module as p3m
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fft

time = np.arange(0,5,1/100)

activity_1, activity_2, activity_3, activity_4 = p3m.load_data('Arai_sit_relax.txt', 'Arai_sit_meditation.txt', 'Arai_sit_puzzles.txt', 'Arai_bounce_yogaball.txt')

concatenated_data = p3m.load_time(time, activity_1, activity_2, activity_3, activity_4)

plt.figure(1, clear=True)

plt.plot(time, concatenated_data)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Concatenated Activities')
plt.legend()

plt.figure(2, clear=True)

plt.plot(time, activity_1)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Sitting at Rest')
plt.legend()

plt.figure(3, clear=True)

plt.plot(time, activity_2)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Relaxing activity that is not rest')
plt.legend()

plt.figure(4, clear=True)

plt.plot(time, activity_3)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Mentally stressful activity')
plt.legend()

plt.figure(5, clear=True)

plt.plot(time, activity_4)
plt.xlabel('Time (s)')
plt.ylabel('Raw Data')
plt.title('Physically stressful activity')
plt.legend()
plt.tight_layout()
plt.show()

