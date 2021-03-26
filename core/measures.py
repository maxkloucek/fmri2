import numpy as np

# from numba import njit


# @njit wont work with njit, what if i used my other less vectorised code?
# this is slow, even though its neat and numpy
def correlations(configurations):
    # configurations = configurations
    B, N = configurations.shape
    si = np.zeros(N)  # <si>
    si_sj = np.zeros(N**2).reshape((N, N))
    # ----- <si> ----- #
    si = np.mean(configurations, axis=0)
    # --- <si><sj> --- #
    si_sj = np.outer(si, si)
    # ---  <sisj> ---- #
    sij = np.matmul(configurations.T, configurations)/B
    return si, sij, si_sj


# @njit
def correlations2(configurations):
    B, N = configurations.shape
    si = np.zeros(N)
    si_sj = np.zeros(N**2).reshape((N, N))
    sij = np.zeros(N**2).reshape((N, N))
    # use a differnet index than t, as this somehow implies time?
    # little n?
    # ----- <si> ----- #
    for i in range(0, N):
        si[i] = configurations[:, i].mean()
    # --- <si><sj> --- #

    # np.mean(configurations,0) # <si>#can I rename this si?
    si_sj = np.outer(si, si)

    # ---  <sisj> ---- #
    for t in range(0, B):
        config = configurations[t]
        sij = sij + np.outer(config, config)
    sij = sij/B
    return si, sij, si_sj


def energy(configuration, **kwargs):
    # helper = Helper(interaction_matrix)
    # try to speed this up by ignoring terms where J == 0#can I use neighbour
    # list in this orginial calc? -> still scales with N^2
    # this is for Ising spins!! (i.e. +-1!)
    # i should use the typed neighbour lists to calculate in here!!!!
    # or is this even helpful, will it just always be slow?
    # just leave it as is for now!!
    interaction_matrix = kwargs['J']
    N = configuration.size
    # neighbour_list = mc.build_typed_neighbour_list(
    #    interaction_matrix, d=2, TH=0)

    upper_indices = np.triu_indices(N, k=1)
    interactions = interaction_matrix[upper_indices]
    config_matrix = np.outer(configuration, configuration)
    config_pairs = config_matrix[upper_indices]  # gives i<j terms
    E = - np.sum(interactions * config_pairs) - \
        np.sum(np.diagonal(interaction_matrix)*configuration)
    return E


def abs_magnetisation(configuration, **kwargs):
    mag = np.mean(configuration)
    mag = abs(mag)
    return mag


def magnetisation(configuration, **kwargs):
    mag = np.mean(configuration)
    return mag


# think I've just discovered that this is a really slow way
# of doing it, but oh well, its nice to have for now!
def trajectory(configurations, observable, **kwargs):
    observable_trajectory = []
    for config in configurations:
        observable_trajectory.append(observable(config, **kwargs))
    return np.array(observable_trajectory)


# this is a very simple error measurement!
def error(J_true, J_traj):
    error = []
    for J_current in J_traj:
        # do I need a / 2 in here, because of symmetry?
        diff = abs(J_true - J_current)
        # diff = J_true - J_current
        # diff = diff**2  # this is the squared difference summed!
        # this depends on N, so I'm not sure this is good
        # think it should be average!
        # sum vs dfiff not sure
        error.append(np.mean(diff))
        # error.append(np.sum(diff))
    return np.array(error)


def self_correlation(spin_trajectory, start_time):
    # f(t-tw) how do I construct this...?
    # make a sort of histogram?
    # the final thing should be a function of deltaT
    # i.e. the delay (tw)
    # spin_trajectory = spin_trajectory[start_time:, :]
    B, N = spin_trajectory.shape
    print(B)
    print('----')
    # x = B
    x = 1
    start_times = np.arange(0, x)  # B!
    # self_corr = []
    self_correlations = np.zeros((x, B))
    self_correlations[self_correlations == 0] = np.nan
    print(self_correlations.shape)
    # start_times = [0]
    for start_time in start_times:
        spin_traj = np.copy(spin_trajectory)
        spin_traj = spin_traj[start_time:, :]
        print(spin_traj.shape)
        B, N = spin_traj.shape
        delays = np.arange(0, B)
        # delays = np.arange(0, 1)
        for delay in delays:
            spins = spin_traj[0, :]
            spins_delayed = spin_traj[delay, :]
            delayed_correlation = (np.mean(spins * spins_delayed))
            self_correlations[start_time, delay] = delayed_correlation
    # I'm not sure this is right yet...

    print(self_correlations)
    means = np.nanmean(self_correlations, axis=0)
    print(means.shape)
        # print(start_time)
        # spin_trajectory = spin_trajectory[start_time:, :]
        # B, N = spin_trajectory.shape
        # delays = np.arange(0, B)
        # print(B)
        # delayed_correlation = np.zeros(B)
        # delayed_correlation[delayed_correlation == 0] = np.nan

            # delayed_correlation[delay] = (np.mean(spins * spins_delayed))
        # self_corr.append(self_corr_t)
        # self_corr.append(delayed_correlation)
    # self_corr = np.array(self_corr)
    # for corr in self_corr:
    #    print(len(corr))
    # print(self_corr.shape)
    # print(self_corr[-1].shape)
    return means
