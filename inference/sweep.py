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
        run_directory, model, metadata, init_config,
        MCCs_eq=100000, MCCs_prod=100000, cycle_dumpfreq=10):
    # have to find some way of naming this! maybe from the metadata?
    # i.e. have some thing that tracks the GD steps? yep thats how
    # I'll do it!
    # lets get rid of reps here too!!!!
    # I don't like this way of doing this with reps, just run indep
    # sims to get repeats!
    N, _ = model.shape
    if init_config is None:
        initial_config = aux.initialise_ising_config(N, 0)
    else:
        initial_config = init_config
        # should do a check for the shape here or whatever!

    eq_traj, eq_energy = mc.simulate(
        model, initial_config, MCCs_eq, cycle_dumpfreq)
    eq_config = eq_traj[-1]

    prod_traj, prod_energy = mc.simulate(
        model, eq_config, MCCs_prod, cycle_dumpfreq)

    fname = join(run_directory, 'c{}r{}trajectory'.format(0, 0))
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
def model_update(trajectory, data_spin_matrix, J_current, learning_rate=0.1):
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
        GD_steps=10,
        MCCs_init_eq=100000, MCCs_anneal_step=100000, cycle_dumpfreq=10):
    model = model_initialisation
    N, _ = model.shape
    model_trajectory = np.zeros((GD_steps, N, N))  # contains updated models!
    print(model_trajectory.shape, model_trajectory[0].shape)
    # I could probably have this bit in the loop, its a bit hacked atm!
    initial_config = aux.initialise_ising_config(N, 0)
    eq_traj, eq_energy = mc.simulate(
            model, initial_config, MCCs_init_eq, cycle_dumpfreq)
    model = model_update(eq_traj, data_spin_matrix, model)
    model_trajectory[0] = model
    initial_config = eq_traj[-1]

    for step in range(1, GD_steps):
        print('GD step: {}'.format(step))
        trajectory_eq, energy_eq = mc.simulate(
            model, initial_config, MCCs_init_eq, cycle_dumpfreq)
        initial_config = trajectory_eq[-1]
        trajectory, energy = mc.simulate(
            model, initial_config, MCCs_anneal_step, cycle_dumpfreq)

        model = model_update(trajectory, data_spin_matrix, model)
        model_trajectory[step] = model
        initial_config = trajectory[-1]
    return model_trajectory


def gradient_descent(
        run_directory, model_initialisation, data_spin_matrix, metadata,
        GD_steps=10, MCCs_eq=100000, MCCs_prod=100000,
        cycle_dumpfreq=10):
    model = model_initialisation
    N, _ = model.shape
    model_trajectory = np.zeros((GD_steps, N, N))  # contains updated models!
    print(model_trajectory.shape, model_trajectory[0].shape)
    # change this so its no eq and only thing
    for step in range(0, GD_steps):
        print(step)
        # this wont work anymore because of the order of stuff!
        trajectory = simple_sim(
            run_directory, model, metadata, None,
            MCCs_eq, MCCs_prod, cycle_dumpfreq)
        si_model, sij_model, si_sj_model = m.correlations(trajectory)
        model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
        model = parameter_update(data_spin_matrix, model_spin_matrix, model)
        model_trajectory[step] = model
    return model_trajectory


# don't give it a way to modify cycle dumpfreq for now!
# set a default initial model!
class IsingInference:
    def __init__(self, data_spin_matrix, output_directory):
        N, _ = data_spin_matrix.shape
        self.dir_out = output_directory
        self.N = N
        self.data_spin_matrix = data_spin_matrix
        # should I have a self here, I think it makes more sense like this!
        # keeps track of current best guess model_spin_matrix
        self.model_spin_matrix = np.NaN
        self.momentum = np.zeros_like(data_spin_matrix)
        # think about how to do this reasonably!
        # self.momentum

    def fit_MonteCarlo(
            self, initial_model, learning_rate,
            GD_steps, MCcycles_eq, MCcycles_production):

        model = initial_model
        model_trajectory = np.zeros((GD_steps, self.N, self.N))
        initial_config = aux.initialise_ising_config(self.N, 0)
        diffs = []
        for step in range(0, GD_steps):
            print('GD step: {}'.format(step))
            trajectory_eq, energy_eq = mc.simulate(
                model, initial_config, MCcycles_eq)
            initial_config = trajectory_eq[-1]
            trajectory, energy = mc.simulate(
                model, initial_config, MCcycles_production)

            model = self.model_update(
                trajectory, model, 0.05, step, GD_steps)
            model_trajectory[step] = model
            initial_config = trajectory[-1]
            diffs.append(self.convergence_check())
        return model_trajectory, diffs

    def save(self):
        return 0

    # returns split and flatrned hs and Js for further analysis!
    def split_diagonal(self, model):
        N, _ = model.shape
        hs = np.diagonal(model)
        Js = model[np.triu_indices(N, k=1)]
        return hs, Js

    def convergence_check(self):
        si_data, sij_data = self.split_diagonal(self.data_spin_matrix)
        si_model, sij_model = self.split_diagonal(self.model_spin_matrix)
        observables_data = np.append(si_data, sij_data)
        observables_model = np.append(si_model, sij_data)
        difference = observables_data - observables_model
        # mean_diff = np.mean(difference)
        diff = np.linalg.norm(difference)
        return diff

    def fit_nMF(self):
        return 0

    def model_update(
            self, trajectory, J_current, learning_rate, step, GD_steps):
        si_model, sij_model, si_sj_model = m.correlations(trajectory)
        self.model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
        dS = self.data_spin_matrix - self.model_spin_matrix
        # maybe I need a smaller learning rate!
        # pass this as a dictonary?
        learning_rate = self.learning_scheme(
            'constant', learning_rate, step, GD_steps)
        J_new = J_current + learning_rate * dS
        return J_new

    def learning_scheme(self, scheme_choice, scheme_parameter, step, GD_steps):
        if scheme_choice == 'constant':
            alpha = scheme_parameter
            return alpha
        if scheme_choice == 'VarInc':
            alpha_init = 0.3
            alpha_fin = 0.01
            alphas = np.linspace(alpha_init, alpha_fin, GD_steps)
            # tau = 2  # (i.e. half it each time)
            # scheme_parameter = 5
            # section_splits = GD_steps / scheme_parameter
            # current_section = step / section_splits
            # power = round(current_section)
            # print(power)
            # alpha = alpha_init * ((1/tau) ** power)
            alpha = alphas[step]
            print(alpha)
            return alpha


class BoltzmannLearning:
    def __init__(self, data_spin_matrix, output_directory):
        N, _ = data_spin_matrix.shape
        self.dir_out = output_directory
        self.N = N
        self.data_spin_matrix = data_spin_matrix
        # should I have a self here, I think it makes more sense like this!
        # keeps track of current best guess model_spin_matrix
        self.model_spin_matrix = np.NaN
        self.momentum = np.zeros_like(data_spin_matrix)
        # think about how to do this reasonably!
        # self.momentum

    def fit_MonteCarlo(
            self, initial_model, learning_rate,
            GD_steps, MCcycles_eq, MCcycles_production):

        model = initial_model
        model_trajectory = np.zeros((GD_steps, self.N, self.N))
        initial_config = aux.initialise_ising_config(self.N, 0)
        diffs = []
        for step in range(0, GD_steps):
            print('GD step: {}'.format(step))
            trajectory_eq, energy_eq = mc.simulate(
                model, initial_config, MCcycles_eq)
            initial_config = trajectory_eq[-1]
            trajectory, energy = mc.simulate(
                model, initial_config, MCcycles_production)

            model = self.model_update(
                trajectory, model, 0.05, step, GD_steps)
            model_trajectory[step] = model
            initial_config = trajectory[-1]
            diffs.append(self.convergence_check())
        return model_trajectory, diffs

    def save(self):
        return 0

    # returns split and flatrned hs and Js for further analysis!
    def split_diagonal(self, model):
        N, _ = model.shape
        hs = np.diagonal(model)
        Js = model[np.triu_indices(N, k=1)]
        return hs, Js

    def convergence_check(self):
        si_data, sij_data = self.split_diagonal(self.data_spin_matrix)
        si_model, sij_model = self.split_diagonal(self.model_spin_matrix)
        observables_data = np.append(si_data, sij_data)
        observables_model = np.append(si_model, sij_data)
        difference = observables_data - observables_model
        # mean_diff = np.mean(difference)
        diff = np.linalg.norm(difference)
        return diff

    def model_update(
            self, trajectory, J_current, learning_rate, step, GD_steps):
        si_model, sij_model, si_sj_model = m.correlations(trajectory)
        self.model_spin_matrix = aux.gen_spin_matrix(si_model, sij_model)
        dS = self.data_spin_matrix - self.model_spin_matrix
        # maybe I need a smaller learning rate!
        # pass this as a dictonary?
        learning_rate = self.learning_scheme(
            'constant', learning_rate, step, GD_steps)
        J_new = J_current + learning_rate * dS
        return J_new

    def learning_scheme(self, scheme_choice, scheme_parameter, step, GD_steps):
        if scheme_choice == 'constant':
            alpha = scheme_parameter
            return alpha
        if scheme_choice == 'VarInc':
            alpha_init = 0.3
            alpha_fin = 0.01
            alphas = np.linspace(alpha_init, alpha_fin, GD_steps)
            # tau = 2  # (i.e. half it each time)
            # scheme_parameter = 5
            # section_splits = GD_steps / scheme_parameter
            # current_section = step / section_splits
            # power = round(current_section)
            # print(power)
            # alpha = alpha_init * ((1/tau) ** power)
            alpha = alphas[step]
            print(alpha)
            return alpha