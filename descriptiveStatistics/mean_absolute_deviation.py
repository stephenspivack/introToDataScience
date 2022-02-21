# This function computes the M.A.D. for input 'data'
import numpy as np
def mean_absolute_deviation_func(data):
    M = np.mean(data)
    sum = 0
    for ii in range(len(data)):
        dev = np.absolute(data[ii] - M)
        sum = sum + dev
    mad = sum/len(data)
    return mad # you mad bro?