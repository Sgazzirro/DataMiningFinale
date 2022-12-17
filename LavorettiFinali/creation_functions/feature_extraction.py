import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from numpy.fft import fft
import detecta

from scipy.signal import savgol_filter


def noise_filter(dataframe):
    for column in dataframe.columns:
        dataframe[column] = savgol_filter(dataframe[column], 5, 2, axis=0)
    return dataframe


def compute_peaks(data):
    X=fft(data)
    N=len(X)
    n=np.arange(N)
    sr=1/50
    T=N/sr
    freq=n/T
    c=20

    n_oneside=N//2
    x=freq[1:n_oneside]
    y=np.abs(X[1:n_oneside])

    max_peak_height=np.amax(y)/c
    peaks=[]
    while len(peaks)<5:
        peaks=detecta.detect_peaks(y, mph=max_peak_height)
        c+=10
        if c>=100:
            break
        max_peak_height=np.amax(y)/c

    peaks_x=peaks/T
    peaks_y=y[peaks]

    if(len(peaks_x)<5):
        for i in range(0,5-len(peaks_x)):
            peaks_x=np.append(peaks_x, np.nan)
            peaks_y=np.append(peaks_y, np.nan)

    return peaks_x[0:5], peaks_y[0:5]


def find_fft_points(data, name):
    (indices_peaks, peaks) = compute_peaks(data)
    columns_x=[name + "X#1", name + "X#2", name + "X#3", name + "X#4", name + "X#5"]
    columns_y=[name + "P#1", name + "P#2", name + "P#3", name + "P#4", name + "P#5"]
    x_p = pd.DataFrame(data=indices_peaks).T
    x_p.columns = columns_x
    y_p = pd.DataFrame(data=peaks).T
    y_p.columns = columns_y
    tot_p = pd.concat([x_p, y_p], axis=1)
    return tot_p


def compute_time_features(df):
    data_total = pd.DataFrame()
    for column in df.columns:
        temp = find_time_features(df[column], column)
        data_total = pd.concat([data_total, temp], axis=1)
    return data_total


def find_time_features(data, name):
    columns = [name + "_mean", name + "_std", name + "_range", name + "_IRQ", name + "_kurtosis", name + "_skewness"]
    properties = [np.mean(data), np.std(data), np.max(data) - np.min(data),
                  np.quantile(data, 0.75) - np.quantile(data, 0.25), kurtosis(data), skew(data)]
    d = pd.DataFrame(data=properties).T
    d.columns = columns
    return d


def compute_freq_features(df):
    data_total = pd.DataFrame()
    for column in df.columns:
        temp = find_fft_points(df[column], column)
        data_total = pd.concat([data_total, temp], axis=1)
    return data_total


def collapse(df):
    time_df=compute_time_features(df)
    freq_df=compute_freq_features(df)
    return pd.concat([time_df, freq_df], axis=1)


def extract_features(dataframe, sample_number):
    i = dataframe.shape[0]//sample_number
    j=0
    filtered_df=noise_filter(dataframe)
    df_time_series=pd.DataFrame()
    for count in range(1,i):
        samples_df=filtered_df.iloc[j:sample_number*count, :]
        new_df=collapse(samples_df)
        if(count==1):
            df_time_series=new_df
        else:
            df_time_series = pd.concat([df_time_series, new_df], axis=0)
        j=sample_number*count
    return df_time_series
