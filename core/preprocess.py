import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from . import measures as m
from . import aux


def load_fmri(
        data_dir='/Users/mk14423/Desktop/code/0-fmri/data-Kajimura',
        day='01'):
    filename = data_dir + '/ROISignals_Sub_' + day + '.mat'
    mat_contents = scipy.io.loadmat(filename)
    data = mat_contents['ROISignals']  # [:, 0:5]
    z, S, mu, sigma = binarize(data, 0)

    return data, z, S
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


def get_MMconditions():
    d1s = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    days_MM = [
        '27', '28', '31', '32', '34', '37', '39', '40', '42', '43', '44', '47',
        '48', '50', '51', '55', '56', '58']
    int_days_MM = np.array([int(i) for i in days_MM])
    int_days_noMM = []
    c = 0
    for i in range(0, 59):
        min_val = np.min(int_days_MM)
        if i == min_val:
            int_days_MM[c] = 100
            c += 1
        else:
            int_days_noMM.append(i)
    days_noMM = d1s
    days_noMM.extend([str(j) for j in int_days_noMM[10:]])
    days_all = days_MM + days_noMM
    days_all = sorted(days_all, key=lambda x: int(x))
    return days_noMM, days_MM, days_all


# retruns spin matrix averaged over given set of days
def average_spin_matrix(days, i=0, j=0):
    spin_matricies = []
    for day in days:
        data, z, s_trajectory = load_fmri(day=day)
        si, sij, si_sj = m.correlations(s_trajectory)
        spin_matrix = aux.gen_spin_matrix(si, sij)
        spin_matricies.append(spin_matrix)
    spin_matricies = np.array(spin_matricies)

    # obs = []
    fig, ax = plt.subplots(3, 3)
    ax = ax.ravel()
    for c, matrix in enumerate(spin_matricies[0:9]):
        ax[c].imshow(matrix)
        # obs.append(matrix[i, j])
    # plt.hist(obs, 25)
    plt.show()
    avrg_spin_matrix = np.mean(spin_matricies, axis=0)
    return avrg_spin_matrix
