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


def trajectory(configurations, observable, **kwargs):
    observable_trajectory = []
    for config in configurations:
        observable_trajectory.append(observable(config, **kwargs))
    return np.array(observable_trajectory)
