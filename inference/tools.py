# general use functions that I might like to use anywhere
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))


# return mu, & sigma + pval of histogrammed data?
# input binned data!
def gaussian_fit(xs, ys, initialisation=np.array([1, 1, 1])):
    optimal_parameters, pcov = curve_fit(gaussian, xs, ys, p0=initialisation)
    yfit = np.array([gaussian(x, *optimal_parameters) for x in xs])
    residuals = ys - yfit
    return yfit, optimal_parameters, residuals
    # return the y values as an array! that's what i want from this!


# k-means function next?! YEP!
def kmeans(data, n=2):
    # data is raw trajectory data -> 1D! (NOT BINNED)
    data = data.reshape(-1, 1)
    label_indicators = np.arange(0, n)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
    means = kmeans.cluster_centers_
    means = means.ravel()

    labels = kmeans.labels_

    split_data = [data[labels == li].ravel() for li in label_indicators]
    split_data = np.array(split_data, dtype=object)
    # N
    N_cluster = [data.size for data in split_data]
    # print(N_cluster)
    # cluster borders?
    cluster_mins = [np.min(data) for data in split_data]
    cluster_maxs = [np.max(data) for data in split_data]
    # print(cluster_mins, cluster_maxs)
    # metadata_dictionary
    metadata = {
        "means": means,
        "mins": cluster_mins,
        "maxs": cluster_maxs,
        "Nconstituents": N_cluster
    }
    return split_data, metadata


def odd_even_check(number):
    if number % 2:
        odd = True  # pass # Odd
    else:
        odd = False  # pass # Even
    return odd


# binned ftt of multiple time series of the same length
# should be shape (Number of series, Number of time points per series)
# could put this into tools! should be shape(Nseries, Ntimepoints)
def fourier_transform_multi_series(trajectories):
    trajectories = np.array(trajectories)
    Nsamples, Nseries = trajectories.shape

    odd = odd_even_check(Nseries)
    # selecting only +ve frequency terms
    if odd is True:
        Ncut = (Nseries - 1) / 2
    elif odd is False:
        Ncut = int((Nseries / 2) - 1)
    else:
        print('You dun goofed')
        return 0
    frequencies = fftfreq(Nseries)[0: Ncut]

    spectra = np.empty((Nsamples, frequencies.size))
    # spec_histogram = np.zeros_like(frequencies)
    # from scipy.signal import hann as window
    for i, trajectory in enumerate(trajectories):
        # w = window(Nseries)
        spectrum = np.abs(fft(trajectory))[0: Ncut]
        # spectrum = 2.0 / Nseries * spectrum  # this is in example; why?
        # spec_histogram += spectrum
        # why am I including 0? shouldn't it start from 1??!?
        # I DON'T KNOW WHATS GOING ON!
        spectra[i] = spectrum

    # spec_histogram /= Nsamples
    spec_means = np.mean(spectra, axis=0)
    spec_std = np.std(spectra, axis=0)
    return frequencies, spec_means, spec_std
