import json
import numpy as np
# import matplotlib as pl
import matplotlib.pyplot as plt

import inference.core.io as io

from os.path import join
from os.path import isfile
from sklearn.cluster import KMeans


def get_metadata(run_dir):
    meta_path = join(run_dir, '_metadata.json')
    with open(meta_path, 'r') as fp:
        metadata = json.load(fp)
    return metadata


def plotHist(run_dirs, nbins=100):
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    for run_dir in run_dirs:
        md = get_metadata(run_dir)
        sweep_parameter = np.array(md["SweepParameterValues"])
        N = md["SystemSize"]
        for i, param in enumerate(sweep_parameter):
            print(param)
            fname = run_dir + 'c{}r0trajectory.npz'.format(i)
            with open(fname, 'rb') as fin:
                traj = np.load(fin)
                eq = traj['eq_traj']
                prod = traj['prod_traj']

            meq = np.mean(eq, axis=1)
            mprod = np.mean(prod, axis=1)
            m = np.concatenate((meq, mprod))
            mbins = np.linspace(-1, 1, nbins)
            msqrbins = np.linspace(0, 1, round(nbins / 2))
            ax[0].hist(m, bins=mbins)
            ax[1].hist(m ** 2, bins=msqrbins)
            # ax.plot(m, '.')
            # ax.set(title='T = {}'.format(Ts[state]))
    plt.legend()
    plt.show()


def traj(run_dir, statepoints, temps):
    fig, ax = plt.subplots(2, round((statepoints.size) / 2), figsize=(16, 8))
    ax = ax.ravel()
    for i, state in enumerate(statepoints):
        print(state)
        fname = run_dir + 'c{}r0trajectory.npz'.format(state)
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            # eq = traj['eq_traj']
            prod = traj['prod_traj']

        mprod = np.mean(prod, axis=1)
        m = mprod

        ax[i].plot(m, '.')
        ax[i].plot(m**2, '.')
        ax[i].set(title='T = {:.2f}'.format(Ts[state]))
    plt.show()


'''
def magchi(run_dirs, start=0, window=0):
    cols = pl.cm.viridis(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    twinned_ax = ax.twinx()
    ax.set(xlabel='T', ylabel=r'$|m|$')
    twinned_ax.set(ylabel=r'$\chi$')
    for run_dir in run_dirs:
        ms = []
        chis = []
        md = get_metadata(run_dir)
        N = md["SystemSize"]
        temps = np.array(md["SweepParameterValues"])
        statepoints = np.arange(0, temps.size)

        for i, state in enumerate(statepoints):
            print(state)
            fname = run_dir + 'c{}r0trajectory.npz'.format(state)
            with open(fname, 'rb') as fin:
                traj = np.load(fin)
                prod = traj['prod_traj']
            # it should work if I replace mean with sum right?
            # think so!
            mprod = abs(np.sum(prod, axis=1))
            var = np.var(mprod)
            ms.append(abs(np.mean(mprod)))
            chis.append(var / temps[i])

        l1, = ax.plot(
            temps, np.array(ms) / N,
            'o', color=cols[0], label=r'$|m|$')
        l2, = twinned_ax.plot(
            temps, chis,
            'v', color=cols[1], label=r'$\chi$')

    lines = [l1, l2]
    ax.legend(lines, [line.get_label() for line in lines])
    plt.show()


def ECv(run_dirs, start=0, window=0):
    cols = pl.cm.viridis(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    twinned_ax = ax.twinx()
    ax.set(xlabel='T', ylabel=r'$E / N$')
    twinned_ax.set(ylabel=r'$C _{v}$')
    for run_dir in run_dirs:
        Edensities = []
        Cvs = []
        md = get_metadata(run_dir)
        temps = np.array(md["SweepParameterValues"])
        statepoints = np.arange(0, temps.size)

        for i, state in enumerate(statepoints):
            print(state, temps[state])
            fname = run_dir + 'c{}r0trajectory.npz'.format(state)
            with open(fname, 'rb') as fin:
                traj = np.load(fin)
                energy_trajectory = np.array(traj['prod_E']) * temps[state]
            E, Edensity, Efluc = observable_and_fluctuation(
                energy_trajectory, md)

            Edensities.append(Edensity)
            Cvs.append(Efluc / (temps[state] ** 2))
        l1, = ax.plot(
            temps, Edensities,
            'o', color=cols[0], label=r'$\frac{E}{N}$')
        l2, = twinned_ax.plot(
            temps, Cvs,
            'v', color=cols[1], label=r'$C_ {v}$')

    lines = [l1, l2]
    ax.legend(lines, [line.get_label() for line in lines])
    plt.show()
'''


# add an if qOP=True parameter!
def plotObsAndFluc(run_dirs, recalc=False, spin_glassOP=False):
    # Tc = 2.269J/kb!
    # think I want to call it recalc!
    lM = r'<$ |M| / N $>'
    lChi = r'$\chi ^{*} / N$'
    lE = r'<$E / N$>'
    lCv = r'$C_ {v} / N$'

    labels = [lM, lChi, lE, lCv]
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    for a, l in zip(ax.ravel(), labels):
        a.set(xlabel='T', ylabel=l)
        if spin_glassOP is False:
            a.axvline(2.269, c='k')

    for run_dir in run_dirs:
        print(run_dir)
        md = get_metadata(run_dir)
        Ts = np.array(md["SweepParameterValues"])
        N = md["SystemSize"]
        precalc_file = run_dir + 'analysis_ECvMChi.npz'
        if recalc is True:
            obsDensities, obsFlucs = ECvMChi_calc(run_dir)
        else:
            if isfile(precalc_file) is True:
                with open(precalc_file, 'rb') as fin:
                    arrays = np.load(fin)
                    obsDensities = arrays["Obs"]
                    obsFlucs = arrays["Fluc"]
            else:
                obsDensities, obsFlucs = ECvMChi_calc(run_dir)

        # MAKE SURE YOU / N TO MAKE COMPARABLE!!
        # ah I think my problem is that ealier I was
        # recording E / N so its too mcuh now?
        # try timsing by N just to see, and then rerecording this data!
        # yep this was my problem, hurray :D:D:D
        ax[0, 0].plot(Ts, obsDensities[0], '.', label='N = {}'.format(N))
        ax[0, 1].plot(Ts, obsFlucs[0] / N, '.')
        ax[1, 0].plot(Ts, obsDensities[1], '.')
        ax[1, 1].plot(Ts, obsFlucs[1] / N, '.')
        if spin_glassOP is True:
            ms, qs = sgOP_calc(run_dir)
            ax[0, 0].plot(Ts, abs(ms), marker='.', label='m')
            ax[0, 0].plot(Ts, qs, marker='.', label='q')
    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()


def sgOP_calc(run_dir):
    md = get_metadata(run_dir)
    N = md["SystemSize"]
    sweep_parameters = np.array(md["SweepParameterValues"])
    ms = np.zeros(sweep_parameters.size)
    qs = np.zeros(sweep_parameters.size)
    for i, sp in enumerate(sweep_parameters):
        fname = run_dir + 'c{}r0trajectory.npz'.format(i)
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            config_traj = traj['prod_traj']
        sis = np.mean(config_traj, axis=0)
        sis_sqr = sis ** 2
        m = np.mean(sis)
        q = np.mean(sis_sqr)
        ms[i] = m
        qs[i] = q
        # print(config_traj.shape)
    return ms, qs


def ECvMChi_calc(run_dir):
    md = get_metadata(run_dir)
    Ts = np.array(md["SweepParameterValues"])
    obsDensities = np.zeros((2, Ts.size))
    obsFlucs = np.zeros((2, Ts.size))
    for i, T in enumerate(Ts):
        print(T)
        fname = run_dir + 'c{}r0trajectory.npz'.format(i)
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            # E_traj = traj['prod_E'] * sweep_parameter * N
            E_traj = traj['prod_E'] * T
            config_traj = traj['prod_traj']

        M_traj = np.sum(config_traj, axis=1)
        Mabs_traj = abs(M_traj)
        Mabs, MabsDensity, MabsFluc = observable_and_fluctuation(
            Mabs_traj, md)
        obsDensities[0, i] = MabsDensity
        obsFlucs[0, i] = MabsFluc / T

        E, EDensity, EFluc = observable_and_fluctuation(E_traj, md)
        obsDensities[1, i] = EDensity
        obsFlucs[1, i] = EFluc / (T ** 2)
    io.save_npz(
        join(run_dir, 'analysis_ECvMChi'), Obs=obsDensities, Fluc=obsFlucs)
    return obsDensities, obsFlucs


def observable_and_fluctuation(observale_trajectory, metadata):
    N = metadata["SystemSize"]
    obs = np.mean(observale_trajectory)
    obs_density = obs / N
    fluct = np.var(observale_trajectory)

    return obs, obs_density, fluct

# dont worry about the autocorr for now
# setup SK next! Then do GD
# finally gonna get somehwere with this!!
# woooo :D
# + scaling analysis


def plotAutoCorr(run_dirs):
    for run_dir in run_dirs:
        md = get_metadata(run_dir)
        Ts = np.array(md["SweepParameterValues"])
    for i, T in enumerate(Ts):
        fname = run_dir + 'c{}r0trajectory.npz'.format(i)
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            config_traj = traj['prod_traj']
        print(config_traj.shape)
        print(config_traj[0].shape)
        corr = np.corrcoef(config_traj[0], config_traj[-1])
        print(corr)

    # md = get_metadata(run_dir)
    return 0


def trajectoryE(run_dir, statepoints, temps):
    fig, ax = plt.subplots(2, round((statepoints.size) / 2), figsize=(16, 8))
    ax = ax.ravel()
    for i, state in enumerate(statepoints):
        print(state)
        fname = run_dir + 'c{}r0trajectory.npz'.format(state)
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            eq = traj['eq_E']
            prod = traj['prod_E']
        print(eq.shape, prod.shape)
        energy_trajectory = np.append(eq, prod)

        ax[i].plot(energy_trajectory / 400, '.')
        ax[i].set(title='T = {:.2f}'.format(temps[state]))
    plt.show()


def k_means_fit(data, n=2):
    data = data.reshape(-1, 1)
    label_indicators = np.arange(0, n)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
    means = kmeans.cluster_centers_
    means = means.ravel()

    labels = kmeans.labels_

    split_data = [data[labels == li].ravel() for li in label_indicators]
    split_data = np.array(split_data)
    return means, split_data
