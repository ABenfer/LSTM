import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import config

print("ver data import = 0.3.3")


def load_interpolated_data():
    sims_names = np.sort([el for el in os.listdir(config.SIMS_INTERP_PATH) if el.endswith('.json')])[:config.N_DATA_POINTS]
    time_series_keys = ["t", "x", "y", "z", "fx", "fy", "fz"]
    time_serieses = np.array([np.genfromtxt(config.TIME_SERIES_INTERP_PATH + name, delimiter=',') for name in time_series_names])

    time = time_serieses[:,0,:]
    x_time_series = time_serieses[:,:,:]
    # substract from every value of x_time_series[:,8:,:] the corresponding value of x_time_series[:,8:,0]
    # for i in [7,8,9]:
    #     x_time_series[:,i,:] = x_time_series[:,i,:] - (np.ones((x_time_series[:,i,:].shape[1], x_time_series[:,i,:].shape[0])) * x_time_series[:,i,0]).T

    # import real features
    real_features_names = np.sort([el for el in os.listdir(config.REAL_FEATURES_PATH) if el.endswith('.txt')])[:config.N_DATA_POINTS]
    real_features_keys = ["d1", "d2", "d3", "k1", "k2", "k3", "q01", "q02", "q03"]   

    real_features = np.array([np.genfromtxt(config.REAL_FEATURES_PATH + name, delimiter=',') for name in real_features_names])
    time_step = np.ones(real_features.shape[0]) * (time[0,1] - time[0,0])
    x_real_features = np.concatenate((real_features[:,-3:] , time_step.reshape(-1,1)), axis=1)
    y_real_features = real_features[:,:-3]

    # print all shapes with names as a table and sanity check
    print("time.shape: ", time.shape)
    print("x_time_series.shape: ", x_time_series.shape)
    print("x_real_features.shape: ", x_real_features.shape)
    print("y_real_features.shape: ", y_real_features.shape)

    for i in range(10):
        print(time_series_keys[i], x_time_series[0, i, 30:34])

    # plot a random sample use random choice
    sample = np.random.randint(0, x_time_series.shape[0])
    plot_data(x_time_series[sample], x_real_features[sample], y_real_features[sample],
                time_series_keys, real_features_keys, real_features_keys)
    
    # check if time series and real features have the same number of samples
    assert x_time_series.shape[0] == x_real_features.shape[0]
    
    #check if real features and time series are the same
    for i in range(len(real_features_names)): 
        assert time_series_names[i].replace("ts_", "").replace(".csv", "") == real_features_names[i].replace("rf_", "").replace(".txt", "")

    return time, x_time_series, x_real_features, y_real_features


def plot_data(x_time_series_sample, x_real_features_sample, y_real_features_sample, 
              x_time_series_keys, x_real_features_keys, y_real_features_keys):
    print(x_time_series_sample.shape)
    # plot time series sample in three subplots
    # first subplot of x: x_time_series_sample[4:7] and y: x_time_series_sample[0]
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(x_time_series_sample[0], x_time_series_sample[4], label=x_time_series_keys[4])
    axs[0].plot(x_time_series_sample[0], x_time_series_sample[5], label=x_time_series_keys[5])
    axs[0].plot(x_time_series_sample[0], x_time_series_sample[6], label=x_time_series_keys[6])
    axs[0].set_title("Input Force")
    axs[0].legend()

    # # second subplot of x: x_time_series_sample[7:10] and y: x_time_series_sample[0]
    # axs[1].plot(x_time_series_sample[0], x_time_series_sample[7], label=x_time_series_keys[7])
    # axs[1].plot(x_time_series_sample[0], x_time_series_sample[8], label=x_time_series_keys[8])
    # axs[1].plot(x_time_series_sample[0], x_time_series_sample[9], label=x_time_series_keys[9])
    # axs[1].set_title("Joint Angle Displacement")
    # axs[1].legend()

    # third subplot of x: x_time_series_sample[1:4] and y: x_time_series_sample[0]
    axs[1].plot(x_time_series_sample[0], x_time_series_sample[1], label=x_time_series_keys[1])
    axs[1].plot(x_time_series_sample[0], x_time_series_sample[2], label=x_time_series_keys[2])
    axs[1].plot(x_time_series_sample[0], x_time_series_sample[3], label=x_time_series_keys[3])
    axs[1].set_title("Displacement")
    axs[1].legend()

    # show plot
    plt.show()
