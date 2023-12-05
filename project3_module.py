def load_time(filenames):
  # takes in array of filenames to load
  # loads into array, returns it for plotting

  return time_array

def load_freq(filenames):
  # takes in array of filenames to load
  # loads into array, returns for plotting

  return freq_array





def load_data(filename_1, filename_2 = "", filename_3 = "", filename_4 = "", plot = True, time=True, freq=False, power=False):
  # we will load the data and then plot it. It will be able to read in files of .csv, .txt, .npz
  # it will also be able to plot time or frequency domain (with optional power) if the data is in either domain.
