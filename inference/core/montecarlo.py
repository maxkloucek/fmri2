import numpy as np
from numba import njit
from numba.typed import List

# import measures as m
# aha! This works!?! Don't love it though!!
from . import measures as m
# from . import aux


# @njit
def build_neighbour_list(interaction_matrix, d, TH):
    # this is to make sure self isn't included in neighbour list keeping,
    # as self interaction is different to pair interaction!
    # looks like it doesnt really always work!
    no_particles, _ = interaction_matrix.shape
    J_neighbours = np.copy(interaction_matrix)
    np.fill_diagonal(J_neighbours, 0)
    # does this work for h=!0? Need to check & be careful I think?
    nonzero_list = np.argwhere(np.abs(J_neighbours) > TH)
    neighbour_list = [[] for _ in range(no_particles)]
    neighbour_length, _ = nonzero_list.shape
    for i in range(0, neighbour_length):
        list_index = nonzero_list[i, 0]
        neighbour_list[list_index].append(nonzero_list[i, 1])
    return neighbour_list


# @njit
def build_typed_neighbour_list(interaction_matrix, d=2, TH=0):
    # lets set a dynmaic threshodl!? How exactly to cope?
    # setting some resoltuion relative to the max where we count
    # it as essentailly being zero? so this is set at 1% of max
    # I feel like 10% of max is too extreme, but lets try it anyway
    # TH = np.max(abs(interaction_matrix)) * 0.01
    # print(TH)
    neighbour_list = build_neighbour_list(interaction_matrix, d, TH)
    typed_neighbour_list = List()
    for i in range(len(neighbour_list)):
        new_list = List()
        for j in range(len(neighbour_list[i])):
            new_list.append(neighbour_list[i][j])
        typed_neighbour_list.append(new_list)
    return typed_neighbour_list


# maybe I have a function called setup, that then calles the simulate function
# def simulate(T, config, N, initial_energy, interaction_matrix,
# neighbour_list, eq_steps, measure_steps, dump_freq):
# kwargs not supported by numba,
# but I can somehow package this up nice and freindyl :)?
# @njit
# setting T = 1 & including it in Jij should work just fine!!
# yep I think I need a wrapper and another function!
# set up in the wrapper, then call my decorated fast function!!


@njit
def sim(
        int_matrix, init_config, init_energy, neighbour_list,
        tot_steps, dump_freq):

    no_particles = init_config.size
    config = np.copy(init_config)
    E = init_energy
    rand_nos = np.random.random(tot_steps)
    trial_indicies = np.random.randint(0, no_particles, tot_steps)
    T = 1  # do it explictly for now
    # configs = []
    # energy = []
    no_sampled_points = int(tot_steps / dump_freq)
    # print(no_sampled_points)
    # let's create them as np arrays of the correct size first & fill them in!
    configs = np.empty((no_sampled_points, no_particles))
    energy = np.empty((no_sampled_points))
    dump_counter = 0
    for i in range(0, tot_steps):
        # print(E / no_particles)
        # this is not every 10 MCCs as it should be, this
        # is simply every 10 steps! Good to know as
        # oversmapling, but not gonna fix my problem! FIXED ALREADY
        if(i % dump_freq == 0):
            # configs.append(np.copy(config))
            # energy.append(E)
            configs[dump_counter, :] = np.copy(config)
            energy[dump_counter] = E
            dump_counter += 1

        trial_index = trial_indicies[i]
        dE = 0
        s_old = config[trial_index]
        s_new = -config[trial_index]
        ds = s_new - s_old

        connected_indices = neighbour_list[trial_index]
        for j in connected_indices:
            dE = dE - (int_matrix[trial_index, j] * config[j] * ds)
        # I think I had this missing for the diagonal!!
        dE = dE - s_new*int_matrix[trial_index, trial_index]
        if (dE / T) < 0:
            config[trial_index] = -config[trial_index]
            E = E + dE
        else:
            if np.exp(-(dE / T)) >= rand_nos[i]:
                config[trial_index] = -config[trial_index]
                E = E + dE
    return configs, energy


def simulate(interaction_matrix, initial_config, mc_cycles, cycle_dumpfreq=10):
    """
    This function performs Metropolis-Hastings Monte-Carlo simulation of a
    configuration of binary valued (+/-1) spins.
    Focus on using J (interaction matrix) as the main input.
    Construct neighbour list should be in here.

    Args:
        interaction_matrix (2D numpy array): The Jij, which contains
        all nformation on the system.
        equilibration_steps (int): Number of steps to equilibrate for
        production_steps (int): Number of steps to produce
        i.e. save configurations from
        initial_config???? needs to match 1st dimension of int matrix!

    Returns:
        configurations (2D np array N x measure_steps / measure_freq): ??
        list: the pairwise sums of ``x`` and ``y``.

    Raises:
        ?

    Examples:
    """
    no_particles, _ = interaction_matrix.shape
    tot_steps = no_particles * mc_cycles
    dump_freq = no_particles * cycle_dumpfreq
    # ah nevermind, tis all good, I've
    # T = 1  # this gets overridden later, so it's all pointless atm
    # this is suuper messy and very concerning!
    # T is encoded in model already, keep in mind that all
    # E in inner function are therefore E/T!!!
    # still have to record the non-spin normalised values!
    # then its all working finally hurray!
    # energy is E over T, remember this! Just cause of how
    # I encode it in the model!
    # there are problems here to do with memory if my sim gets too long!
    # I should really address this!
    initial_energy = m.energy(initial_config, J=interaction_matrix)
    neighbour_list = build_typed_neighbour_list(
                interaction_matrix, d=2, TH=0)  # this needs to be accessable!
    trajectory, energy = sim(
        interaction_matrix, initial_config, initial_energy,
        neighbour_list, tot_steps, dump_freq)
    trajectory = np.array(trajectory)
    # energy = np.array(energy) / no_particles
    # lets return the raw values, not the densities!
    # now I want to save it with a h5py thing
    return trajectory, energy
