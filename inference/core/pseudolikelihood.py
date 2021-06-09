# import sys

import numpy as np
import h5py
import pathlib
import os.path
# import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm import tqdm
from scipy.special import expit
# from os.path import join

from scipy.optimize import minimize
from joblib import Parallel, delayed

import inference.core.analytical as analytical

from inference.io import Readhdf5_mc

h5py.get_config().track_order = True
# coding up the stuff
# waximise the sum over the individaul things.
# " solve N independent gradient-descent problems in N variables"


def logistic_function(x):
    return 1 / (1 + np.exp(-x))


def pseudo_loglikelihoodF(Jtriu, spin=None):
    # add lamba parameter* norm of J sum of abs values
    # (sum aboslute values and multiply by lambda)
    N = spin.shape[1]
    # B = spin.shape[0]
    beta = 1
    # require Jij = Jji
    J = np.zeros((N, N))
    row, col = np.triu_indices(N, k=1)

    J[row, col] = Jtriu
    J += J.T
    # plt.imshow(J)
    # plt.show()
    # note: -log 1/P = log P
    f = np.mean(np.log(
        1+np.exp(
            -2*beta*(spin@J)*spin)
    ),
        axis=0
    )
    # print(f)
    assert len(f) == N, 'Wrong dimensions'
    # print (f.sum())
    # print(Jtriu, Jtriu.min())
    print(f.sum())
    # f = con
    return f.sum()


def PSL_spin_vector(row_index, configuration):
    spin_vector = np.copy(configuration)
    spin_vector *= spin_vector[row_index]
    spin_vector[row_index] = configuration[row_index]
    return spin_vector


# beta explicitly set to 1 everywhere!
def PSL_exponent(row_parameters, row_index, configuration):
    row_spin_combinations = PSL_spin_vector(row_index, configuration)
    exponent = 2 * np.dot(row_spin_combinations, row_parameters)
    return exponent


# this is what I want to minimize for each row!
def PSL_row(row_parameters, row_index, configurations, l1_penalty):
    # so stuff for each sample and then mean over samples
    log_probabilities = []
    for configuration in configurations:
        # SPIN VECTOR:
        # row_spin_combinations = PSL_spin_vector(row_index, configuration)
        # exponent = 2 * np.dot(row_spin_combinations, row_parameters)
        exponent = PSL_exponent(row_parameters, row_index, configuration)
        # print(configuration.shape, configuration)
        # the non-scipy function is waaay quicker, like factor of 2!
        # so just find a way to ignore these errors for now!
        log_probabilities.append(np.log(1 + np.exp(-exponent)))
        # I can probably do this better / quicker!
        # log_probabilities.append(expit(exponent))
    row_pseudo_LL = np.mean(log_probabilities)
    # row_pseudo_LL += l1_penalty * l1reg(row_parameters)
    return row_pseudo_LL


def PSL_row_gradient(row_parameters, row_index, configurations, l1_penalty):
    # grad = 1/(e^x(r) + 1 ) * -2 * beta(=1) * thing
    B, N = configurations.shape
    grad_sums = np.zeros((B, N))
    # print(grad_sums.shape)
    for i, configuration in enumerate(configurations):
        exponent = PSL_exponent(row_parameters, row_index, configuration)
        A = -2 / (np.exp(exponent) + 1)
        row_spin_combinations = PSL_spin_vector(row_index, configuration)
        grad_sums[i] = A * row_spin_combinations
    # print(grad_sums)
    # print(grad_sums.shape)
    # print(np.mean(grad_sums, axis=0))
    gradient_vector = np.mean(grad_sums, axis=0)
    # i.ve turned off l1 penalty for now to supress errors
    # gradient_vector += l1_penalty * l1reg_gradient(row_parameters)
    return gradient_vector


# I need to compare somehow! -> not sure if these are right yet
# or not; haven't properly tested it!
def l1reg(row_parameters):
    return np.sum(np.abs(row_parameters))


def l1reg_gradient(row_parameters):
    return row_parameters / np.abs(row_parameters)

'''
# this function does it for each row indivdually!
def maxPSL(Jguess, configurations, analytical_gradient=False):
    B, N = configurations.shape
    print(Jguess.shape, configurations.shape)
    # x = []
    print(Jguess)
    if analytical_gradient is False:
        gradient_vector = None
    else:
        gradient_vector = PSL_row_gradient
    Jinferred = np.zeros_like(Jguess)
    for spin in range(0, N):
        print('Row {} of {}'.format(spin, N))
        # do the minimisation (giving each row)
        res = minimize(
            PSL_row,
            x0=Jguess[spin],
            args=(spin, configurations),
            jac=gradient_vector,
            method='L-BFGS-B',
            # options={'disp': True, 'maxiter': 20}
        )
        Jinferred[spin] = res.x
        # x.append(PSL_row(res.x, spin, configurations))
    # LL = np.sum(x)
    # print(LL)
    return Jinferred
'''


# should this be a class that has acess to confgiruations?
# this function does it for each row indivdually!
def maxPSL_parallel(
        Jguess, configurations,
        analytical_gradient=False, l1_penalty=0):
    B, N = configurations.shape

    if analytical_gradient is False:
        gradient_vector = None
    else:
        gradient_vector = PSL_row_gradient
    Jinferred = np.zeros_like(Jguess)

    const_args = (configurations, gradient_vector, l1_penalty)
    spins = np.arange(0, N)
    guesses = np.array([guess for guess in Jguess])
    '''
    r = Parallel(n_jobs=-1, verbose=5)(
        delayed(parallel_innerloop)(const_args + (i, j))
        for i, j in zip(spins, guesses))
    '''
    r = Parallel(n_jobs=-1)(
        delayed(parallel_innerloop)(const_args + (i, j))
        for i, j in tqdm(zip(spins, guesses), total=N))

    Jinferred = np.array(r)
    return Jinferred


def parallel_innerloop(inner_args):
    configurations, gradient_vector, l1_penalty, spin, init_guess = inner_args

    res = minimize(
        PSL_row,
        x0=init_guess,
        args=(spin, configurations, l1_penalty),
        jac=gradient_vector,
        method='L-BFGS-B',
        # options={'disp': True, 'maxiter': 20}
    )
    return res.x


# I should write this so it can read anyhing, not just the MC stuff!
# groups many change!
class PLmax:
    def __init__(
            self, fname,
            # contains_multiple_datasets=False,
            dset_label=None,
            file_reader=Readhdf5_mc):
        self.fname = fname
        self.data_directory = pathlib.Path(fname).parent.absolute()
        self.dset_label = dset_label  # keep this for now!
        # delete dset_label?!? or leave the option to pass one!
        # i.e. if dste label = None do blah!
        # how to make this apply alwasy?
        # always call it configurations!
        # this needs some renamining!
        # have an option to chose only one!!
        # needs same keys as before!
        with file_reader(fname) as f:
            self.key_list = f.keys()
            self.data_array = f.read_many_datasets("configurations")
            # should be 3 for multiple and 2 for single
            # now I don't even need the option at the start anymore?
            # just make sure you put single in a list so it works in loop!
            N_dimensions = len(self.data_array.shape)
            # print(len(self.data_array.shape), self.data_array.shape)
            if N_dimensions == 2:
                self.data_array = [self.data_array[0]]
            elif N_dimensions == 3:
                pass
            else:
                raise Exception(
                    ("Invalid Number of datasets; " +
                        "Ndsets={}, only 2 or 3 valid").format(N_dimensions))
            # self.ds = f.read_single_dataset(dset_label, "configurations")
            # print(self.ds.shape)

    def write_to_group(self, inferred_model):
        model_file_path = os.path.join(self.data_directory, 'models.hdf5')
        with h5py.File(model_file_path, 'a') as f:
            dataset = f[self.glabel].create_dataset(
                self.dset_label, data=inferred_model)
            dataset[()] = inferred_model
            # print(list(f['/'].keys()))

    def setup_group(self):
        model_file_path = os.path.join(self.data_directory, 'models.hdf5')
        with h5py.File(model_file_path, 'a') as f:
            key_list = list(f['/'].keys())
            # empty check
            # print(key_list, len(key_list))
            # change this logic somehow!!
            # think it might break, but fix that when you encounter it!
            if not key_list:
                grp_label = 'InferredModels:0'
            elif key_list[0] == 'TrueModels' and len(key_list) < 2:
                grp_label = 'InferredModels:0'
            else:
                grp_label = key_list[-1]
                split = grp_label.split(':')
                grp_label = split[0] + ':' + str(int(split[1]) + 1)
            self.glabel = grp_label
            # print(self.glabel)
            f.create_group(grp_label)

    def infer(
            self, initial_guess_type, analytical_gradient=True, l1_penalty=0):
        self.setup_group()
        for counter, dataset in enumerate(self.data_array):
            s = timer()
            print(
                'Working on Condition: {} of {}'.format(
                    counter + 1, len(self.data_array)))

            self.dset_label = self.key_list[counter]
            self.infer_single_dataset(
                initial_guess_type, dataset, l1_penalty=l1_penalty)
            e = timer()
            print('Inferred in: {:.2f}s'.format(e - s))

    def infer_single_dataset(
            self, initial_guess_type, dataset,
            analytical_gradient=True, l1_penalty=0):
        _, N = dataset.shape
        if initial_guess_type == 'nMF':
            approx = analytical.Approximation(dataset)
            initial_guess = approx.nMF()
        elif initial_guess_type == 'random':
            initial_guess = np.random.rand(N, N)
        else:
            initial_guess = None
        self.p0 = initial_guess

        PLM_model = maxPSL_parallel(
            self.p0, dataset, analytical_gradient, l1_penalty)

        PLM_model = (PLM_model + PLM_model.T) / 2
        self.write_to_group(PLM_model)
        return PLM_model
