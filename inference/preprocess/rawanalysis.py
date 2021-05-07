# doing stuff with the raw data, e.g. histogram + FTT
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import inference.tools as tools

from inference.core import measures as m
from inference.core import aux
from . import readfmri


# retruns spin matrix averaged over given set of days
def average_spin_matrix(days, i=0, j=0):
    spin_matricies = []
    for day in days:
        data, z, s_trajectory = readfmri.load_fmri(day=day)
        si, sij, si_sj = m.correlations(s_trajectory)
        spin_matrix = aux.gen_spin_matrix(si, sij)
        spin_matricies.append(spin_matrix)
    spin_matricies = np.array(spin_matricies)

    # plt.show()
    avrg_spin_matrix = np.mean(spin_matricies, axis=0)
    return avrg_spin_matrix


def histogram(trajectory, nbins=200):
    trajectory = trajectory.ravel()
    fig, ax = plt.subplots(3, 1)
    ax = ax.ravel()

    bins = np.linspace(-8, 8, nbins)
    n, bin_edges, _ = ax[0].hist(
        trajectory, bins=bins,
        # edgecolor='black',
        density=True, alpha=0.2)
    bin_width = bin_edges[1] - bin_edges[0]
    x = (0.5 * bin_width) + bin_edges[:-1]

    nfit, fit_params, residuals = tools.gaussian_fit(x, n)

    ax[0].plot(
        x, nfit, marker=',',
        label="A:{:.2f} mu:{:.2f} sig:{:.2f}".format(
            fit_params[0], fit_params[1], fit_params[2]))
    ax[0].legend()
    ax[0].set_yscale('log')

    Rsquared = r2_score(n, nfit)
    ax[2].hist(
        residuals.ravel(), bins=20,
        label=r'$R^{2}$' + '= {:.3f}'.format(Rsquared))
    ax[2].legend()

    split_data, clustering_metadata = tools.kmeans(trajectory)
    means = clustering_metadata["means"]
    Ns = clustering_metadata["Nconstituents"]
    labels = [
        r'$\mu _{1}=$' + '{:.2f}'.format(means[0]) +
        ' % = {:.2f}'.format(Ns[0] / np.sum(Ns)),
        r'$\mu _{2}=$' + '{:.2f}'.format(means[1]) +
        ' % = {:.2f}'.format(Ns[1] / np.sum(Ns))
    ]
    ax[1].hist(
        split_data, bins, density=True, stacked=True,
        label=labels)
    ax[1].legend()
    plt.show()


def kmeans_thresholds(trajectories):
    ths = []
    # I should really do this differently! -> something doesn't quite work?
    print(trajectories.shape)
    for i in range(0, 399):
        trajectory = trajectories[:, i]
        split_data, clustering_metadata = tools.kmeans(trajectory)
        mins = clustering_metadata['mins']
        maxs = clustering_metadata['maxs']
        # print(mins, maxs)
        threshold = 0.5 * (np.min(maxs) + np.max(mins))
        if abs(threshold) > 0.5:
            print(i, threshold)
            print(maxs, mins)
            print(np.min(maxs), np.max(mins))
        ths.append(threshold)
    return ths
# get the average of min and max
# maybe I do each one individually & make a histogram again?


# this if from https://arxiv.org/pdf/2104.07346.pdf
# trajectory is 1D!
def yushis_eq_test(trajectories, func=np.max):
    trajectory = trajectories
    t_length = trajectory.size  # trajectories[:, 0].size
    T = 236
    N = t_length / T  # somehow round to exclude the last thing
    print(N)
    t_maxs = []
    # for j in range(0, 399):
    #    trajectory = trajectories[:, j]
    for i in range(1, int(N + 1)):
        window = np.array([(i - 1) * T, i * T])
        # print(window)
        sub_traj = trajectory[window[0]: window[1]]
        # print(sub_traj.shape)
        # plt.axvline(x=i * T)
        # given T?
        t_max = np.argmax(sub_traj) / T
        # print(t_max)
        t_maxs.append(t_max)
        # print(j, i)
    t_maxs = np.array(t_maxs)
    print(t_maxs.shape)
    # I GUESS AS tmax CAN ONLY TAKE DISCRETVE VALUES (0<=tmax<N) set bins = T!
    # I'm a bit confused by what time to take and all that Jaz!
    # I think my things are a bit off center, indices and times getting
    # confused
    n, bin_edges = np.histogram(t_maxs, bins=T, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    xs = [bin_start + 0.5 * bin_width for bin_start in bin_edges[: -1]]
    xs = np.array(xs)

    plt.plot(xs, n)
    plt.show()
    plt.hist(t_maxs, bin_edges)
    plt.show()
    return 0


# reshape for FFT analysis
def reshape_multiday_data(multiday_data):
    days, time_points, rois = multiday_data.shape
    # final thing should be shape (ROIs * days, time_points)
    reshaped_data = np.empty((days * rois, time_points))
    for day in range(0, days):
        cut_low = day * rois
        cut_high = (day + 1) * rois
        reshaped_data[cut_low: cut_high] = multiday_data[day].T
    # print(np.array_equal(reshaped_data[0: 399], multiday_data[0].T)) # works
    return reshaped_data
