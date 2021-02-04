import numpy as np
# import matplotlib.pyplot as plt

from os.path import join

from .core import aux
from .core import measures as m
from .core import montecarlo as mc
from .core import io as io


# just save don't return anything now!
# I want an option to initialise the sim differently as well
# eq the slow annelaing, where you use final of last each time!
def parameter_sweep(
        run_directory, models, metadata,
        MCCs_eq=100000, MCCs_prod=100000, reps=1, cycle_dumpfreq=10):

    N, _ = models[0].shape
    for c, model in enumerate(models):
        for rep in range(0, reps):
            initial_config = aux.initialise_ising_config(N, 0)

            eq_traj, eq_energy = mc.simulate(
                model, initial_config, MCCs_eq, cycle_dumpfreq)
            eq_config = eq_traj[-1]
            '''
            fname = join(run_directory, str(c) + 'eq')
            io.save_2Dconfig_image(
                fname, eq_config,
                label=metadata['ParameterName'],
                labelval=metadata['ParameterValues'][c])
            '''
            prod_traj, prod_energy = mc.simulate(
                model, eq_config, MCCs_prod, cycle_dumpfreq)
            '''
            fname = join(run_directory, str(c) + 'fin')
            io.save_2Dconfig_image(
                fname, prod_traj[-1],
                label=metadata['ParameterName'],
                labelval=metadata['ParameterValues'][c])
            '''
            fname = join(run_directory, 'c{}r{}trajectory'.format(c, rep))
            io.save_npz(
                fname,
                eq_traj=eq_traj, eq_E=eq_energy,
                prod_traj=prod_traj, prod_E=prod_energy)
        print(c, metadata['SweepParameterValues'][c])
    return 0


# takes in model, returns the interaction stuff!
def simple_sim(
        run_directory, model, metadata,
        MCCs_eq=100000, MCCs_prod=100000, reps=1, cycle_dumpfreq=10):
    # have to find some way of naming this! maybe from the metadata?
    # i.e. have some thing that tracks the GD steps? yep thats how
    # I'll do it!
    N, _ = model.shape
    for rep in range(0, reps):
        initial_config = aux.initialise_ising_config(N, 0)

        eq_traj, eq_energy = mc.simulate(
            model, initial_config, MCCs_eq, cycle_dumpfreq)
        eq_config = eq_traj[-1]

        prod_traj, prod_energy = mc.simulate(
            model, eq_config, MCCs_prod, cycle_dumpfreq)

        fname = join(run_directory, 'c{}r{}trajectory'.format(0, rep))
        io.save_npz(
            fname,
            eq_traj=eq_traj, eq_E=eq_energy,
            prod_traj=prod_traj, prod_E=prod_energy)
    return prod_traj


def parameter_update(
        data_spin_matrix, model_spin_matrix, J_current, learning_rate=0.1):
    # x = x - alpha (Sd - Sm)
    dS = data_spin_matrix - model_spin_matrix
    # maybe I need a smaller learning rate!
    J_new = J_current + learning_rate * dS
    return J_new


# I could have this bee one whole function, where gen spin matrix
# calls correlations!
def pupdate2(trajectory, data_spin_matrix, J_current, learning_rate=0.1):
    si_model, sij_model, si_sj_model = m.correlations(trajectory)
    model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
    dS = data_spin_matrix - model_spin_matrix
    # maybe I need a smaller learning rate!
    J_new = J_current + learning_rate * dS
    return J_new


# what if I start with a gaussian, is that cheating? Guess so!
# but not raelly because the noise will be all wrong!
# this should get moved to a different file!!
# parameter_sweep should really call simple sweep no?
# maybe I should use kwargs here?!? think so!
# maybe simple sim shouldn't have a run directory?!
# maybe it should, I suppose I still want to save stuff!
def gradient_descent_anneal(
        run_directory, model_initialisation, data_spin_matrix, metadata,
        GD_steps=10, MCCs_init_eq=100000, MCCs_anneal_step=100000,
        reps=1, cycle_dumpfreq=10):
    model = model_initialisation
    N, _ = model.shape
    model_trajectory = np.zeros((GD_steps, N, N))  # contains updated models!
    print(model_trajectory.shape, model_trajectory[0].shape)
    # I could probably have this bit in the loop, its a bit hacked atm!
    initial_config = aux.initialise_ising_config(N, 0)
    eq_traj, eq_energy = mc.simulate(
            model, initial_config, MCCs_init_eq, cycle_dumpfreq)
    model = pupdate2(eq_traj, data_spin_matrix, model)
    model_trajectory[0] = model
    initial_config = eq_traj[-1]
    # si_model, sij_model, si_sj_model = m.correlations(eq_traj)
    # model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
    # model = parameter_update(data_spin_matrix, model_spin_matrix, model)
    # now saving the whole trajectory makes a bit more sense as well!
    for step in range(1, GD_steps):
        print('GD step: {}'.format(step))
        trajectory_eq, energy_eq = mc.simulate(
            model, initial_config, MCCs_init_eq, cycle_dumpfreq)
        initial_config = trajectory_eq[-1]
        trajectory, energy = mc.simulate(
            model, initial_config, MCCs_anneal_step, cycle_dumpfreq)

        model = pupdate2(trajectory, data_spin_matrix, model)
        model_trajectory[step] = model
        initial_config = trajectory[-1]
    return model_trajectory


def gradient_descent(
        run_directory, model_initialisation, data_spin_matrix, metadata,
        GD_steps=10, MCCs_eq=100000, MCCs_prod=100000,
        reps=1, cycle_dumpfreq=10):
    model = model_initialisation
    N, _ = model.shape
    model_trajectory = np.zeros((GD_steps, N, N))  # contains updated models!
    print(model_trajectory.shape, model_trajectory[0].shape)
    # change this so its no eq and only thing
    for step in range(0, GD_steps):
        print(step)
        trajectory = simple_sim(
            run_directory, model, metadata,
            MCCs_eq, MCCs_prod, reps, cycle_dumpfreq)
        si_model, sij_model, si_sj_model = m.correlations(trajectory)
        model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
        model = parameter_update(data_spin_matrix, model_spin_matrix, model)
        model_trajectory[step] = model
    return model_trajectory
