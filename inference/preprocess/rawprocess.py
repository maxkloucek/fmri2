import numpy as np
# import h5py


# this is from: z-values is:
# https://www.biorxiv.org/content/biorxiv/early/2019/07/01/688655.full.pdf
# this is now time dependent maybe I do it like they do in 2014 nature paper!
# this is not about connections anymore now!
# mu and sigma are global values for each t point!
# TH = 0 by necessity sets ~ half on half off?
def binarize(data, TH=0):
    t_len, ROI_len = data.shape
    mu = []  # global signal mean for each time point
    sigma = []  # std of global signal for each time point
    for t in range(0, t_len):
        mu.append(np.mean(data[t, :]))
        sigma.append(np.std(data[t, :]))
    mu = np.array(mu)
    sigma = np.array(sigma)

    z = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    S = np.zeros(t_len*ROI_len).reshape(t_len, ROI_len)
    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            z[t, ROI] = (data[t, ROI] - mu[t]) / sigma[t]

    for t in range(0, t_len):
        for ROI in range(0, ROI_len):
            if z[t, ROI] >= TH:
                S[t, ROI] = 1
            else:  # i.e. < 0
                S[t, ROI] = -1
    return z, S, mu, sigma


def save_h5py_binarized_data():
    return 0
