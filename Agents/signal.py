import numpy as np

def get_macd_signal(fast_array, slow_array):
    fast_mean = np.mean(fast_array)
    slow_mean = np.mean(slow_array)
    return fast_mean - slow_mean
