import numpy as np
# import matplotlib.pyplot as plt
import h5py

from os.path import join
from pathlib import Path

from .core import utils
from .core import measures as m
from .core import montecarlo as mc
from .core import io as io

h5py.get_config().track_order = True


# should I store the models themselves? I guess maybe?
# i imagine they take up a lot of disk space!
# md is dictionary!
class MonteCarlo:
    def __init__(self, run_directory):
        self.run_dir = run_directory

        self.N = None  # system_size
        self.eq_cycles = None
        self.prod_cycles = None
        self.cycle_dumpfreq = None
        self.reps = None
        self.pname = None
        self.pvals = None
        self.md = None

        self.models = None

    def setHyperParameters(
            self,
            eq_cycles=1 * (10 ** 4),
            prod_cycles=5 * (10 ** 4),
            cycle_dumpfreq=10, reps=1):
        self.eq_cycles = eq_cycles
        self.prod_cycles = prod_cycles
        self.cycle_dumpfreq = cycle_dumpfreq
        self.reps = reps
        self.get_metadata()

    # 3D arrays of the interactions in the model
    # first index for each model
    # second two indicies describe said model
    # check square?
    def setSweepModels(self, models):
        models = np.array(models)
        N_sweep_points, system_size, _ = models.shape
        self.N = system_size
        self.models = models
        self.get_metadata()

    # needs to be np array for pvals
    # name and values of sweep parameter
    # descibe the name & values of parameter varied
    # in models
    def describeSweepParameters(self, pname, pvals):
        self.pname = pname
        self.pvals = np.array(pvals)
        self.get_metadata()

    def get_metadata(self):
        if self.pvals is None:
            pvals = 'None'
        else:
            pvals = self.pvals.tolist()
        md = {
            'RunDirectory': self.run_dir,
            'SystemSize': self.N,
            'EqCycles': self.eq_cycles,
            'ProdCycles': self.prod_cycles,
            'CycleDumpFreq': self.cycle_dumpfreq,
            'Repetitions': self.reps,
            'SweepParameterName': self.pname,
            'SweepParameterValues': pvals}
        self.md = md

    def run(self):
        # print(self.run_dir)
        Path(self.run_dir).mkdir(exist_ok=True)
        samples_eq = int(self.eq_cycles / self.cycle_dumpfreq)
        samples_prod = int(self.prod_cycles / self.cycle_dumpfreq)
        samples_tot = samples_eq + samples_prod
        with h5py.File(
                self.md["RunDirectory"] + "/models.hdf5", "w") as f:
            true_model_group = f.create_group("TrueModels")
            for c, model in enumerate(self.models):
                ds_label = (
                        self.md['SweepParameterName'] +
                        '={:.2f}'.format(self.md['SweepParameterValues'][c]))
                ds = true_model_group.create_dataset(ds_label, data=model)
                ds[()] = model

        with h5py.File(
                self.md["RunDirectory"] + "/mc_output.hdf5",
                "w") as f:
            for key, val in self.md.items():
                f.attrs[key] = val

            for c, model in enumerate(self.models):
                for rep in range(0, self.reps):
                    initial_config = utils.initialise_ising_config(self.N, 0)

                    eq_traj, eq_energy = mc.simulate(
                        model, initial_config,
                        self.eq_cycles, self.cycle_dumpfreq)
                    eq_config = eq_traj[-1]

                    prod_traj, prod_energy = mc.simulate(
                        model, eq_config,
                        self.prod_cycles, self.cycle_dumpfreq)

                    group_label = (
                        self.md['SweepParameterName'] +
                        '={:.2f}'.format(self.md['SweepParameterValues'][c]))

                    print(group_label)
                    group = f.create_group(group_label)

                    energy_ds = group.create_dataset(
                        "energy",
                        (samples_tot),
                        compression="gzip")

                    energy_ds[:samples_eq] = eq_energy
                    energy_ds[samples_eq:] = prod_energy

                    config_ds = group.create_dataset(
                        "configurations",
                        (samples_tot, self.N),
                        compression="gzip")

                    config_ds[:samples_eq, :] = eq_traj
                    config_ds[samples_eq:] = prod_traj


def mc_sweep(metadata, models):
    # run_directory, models, metadata,
    # MCCs_eq=100000, MCCs_prod=100000, reps=1, cycle_dumpfreq=10)
    N = metadata["SystemSize"]
    MCCs_eq = metadata["EqCycles"]
    MCCs_prod = metadata["ProdCycles"]
    cycle_dumpfreq = metadata["CycleDumpFreq"]
    samples_eq = int(MCCs_eq / cycle_dumpfreq)
    samples_prod = int(MCCs_prod / cycle_dumpfreq)
    samples_tot = samples_eq + samples_prod
    with h5py.File(
            metadata["RunDirectory"] + "/mc_output.hdf5",
            "w", track_order=True) as f:
        for key, val in metadata.items():
            f.attrs[key] = val

        for c, model in enumerate(models):
            for rep in range(0, metadata["Repetitions"]):
                initial_config = utils.initialise_ising_config(N, 0)

                eq_traj, eq_energy = mc.simulate(
                    model, initial_config, MCCs_eq, cycle_dumpfreq)
                eq_config = eq_traj[-1]

                prod_traj, prod_energy = mc.simulate(
                    model, eq_config, MCCs_prod, cycle_dumpfreq)

                group_label = (
                    metadata['SweepParameterName'] +
                    '={:.2f}'.format(metadata['SweepParameterValues'][c]))

                print(group_label)
                group = f.create_group(group_label)

                energy_ds = group.create_dataset(
                    "energy",
                    (samples_tot),
                    compression="gzip")

                energy_ds[:samples_eq] = eq_energy
                energy_ds[samples_eq:] = prod_energy

                config_ds = group.create_dataset(
                    "configurations",
                    (samples_tot, N),
                    compression="gzip")

                config_ds[:samples_eq, :] = eq_traj
                config_ds[samples_eq:] = prod_traj


def parameter_sweep(
        run_directory, models, metadata,
        MCCs_eq=100000, MCCs_prod=100000, reps=1, cycle_dumpfreq=10):

    N, _ = models[0].shape
    samples_eq = int(MCCs_eq / cycle_dumpfreq)
    samples_prod = int(MCCs_prod / cycle_dumpfreq)
    samples_tot = samples_eq + samples_prod
    with h5py.File(
            run_directory + "/mc_output.hdf5",
            "w", track_order=True) as f:
        for key, val in metadata.items():
            f.attrs[key] = val

        for c, model in enumerate(models):
            for rep in range(0, reps):
                initial_config = utils.initialise_ising_config(N, 0)

                eq_traj, eq_energy = mc.simulate(
                    model, initial_config, MCCs_eq, cycle_dumpfreq)
                eq_config = eq_traj[-1]

                prod_traj, prod_energy = mc.simulate(
                    model, eq_config, MCCs_prod, cycle_dumpfreq)

                group_label = (
                    metadata['SweepParameterName'] +
                    '={:.2f}'.format(metadata['SweepParameterValues'][c]))

                print(group_label)
                group = f.create_group(group_label)

                energy_ds = group.create_dataset(
                    "energy",
                    (samples_tot),
                    compression="gzip")

                energy_ds[:samples_eq] = eq_energy
                energy_ds[samples_eq:] = prod_energy

                config_ds = group.create_dataset(
                    "configurations",
                    (samples_tot, N),
                    compression="gzip")

                config_ds[:samples_eq, :] = eq_traj
                config_ds[samples_eq:] = prod_traj

                # print('----')
                # print(energy_ds.shape, config_ds.shape)
                # group.create_dataset(
                #    "eq-configs", data=eq_traj)[()] = eq_traj
                # group.create_dataset(
                #    "prod-configs", data=eq_traj)[()] = prod_traj
                # group.create_dataset(
                #    "eq-energies", data=eq_energy)[()] = eq_energy
                # group.create_dataset(
                #    "prod-energies", data=eq_energy)[()] = prod_energy
                # theres got to be a better way to do this!
                # full_trajectory = np.vstack((eq_traj, prod_traj))
                '''
                fname = join(run_directory, 'c{}r{}trajectory'.format(c, rep))
                io.save_npz(
                    fname,
                    eq_traj=eq_traj, eq_E=eq_energy,
                    prod_traj=prod_traj, prod_E=prod_energy)
                '''


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
        initial_config = utils.initialise_ising_config(N, 0)
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
    model_spin_matrix = utils.gen_spin_matrix(si_model, sij_model)
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
    initial_config = utils.initialise_ising_config(N, 0)
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
        model_spin_matrix = utils.gen_spin_matrix(si_model, sij_model)
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
        initial_config = utils.initialise_ising_config(self.N, 0)
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
        self.model_spin_matrix = utils.gen_spin_matrix(si_model, sij_model)
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
        initial_config = utils.initialise_ising_config(self.N, 0)
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
        self.model_spin_matrix = utils.gen_spin_matrix(si_model, sij_model)
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
