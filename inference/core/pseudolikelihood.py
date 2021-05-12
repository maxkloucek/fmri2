import numpy as np
import h5py
import pathlib
import os.path
# import matplotlib.pyplot as plt

from timeit import default_timer as timer
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
def PSL_row(row_parameters, row_index, configurations):
    # so stuff for each sample and then mean over samples
    log_probabilities = []
    for configuration in configurations:
        # SPIN VECTOR:
        # row_spin_combinations = PSL_spin_vector(row_index, configuration)
        # exponent = 2 * np.dot(row_spin_combinations, row_parameters)
        exponent = PSL_exponent(row_parameters, row_index, configuration)
        # print(configuration.shape, configuration)
        log_probabilities.append(np.log(1 + np.exp(-exponent)))
    row_pseudo_LL = np.mean(log_probabilities)
    return row_pseudo_LL


def PSL_row_gradient(row_parameters, row_index, configurations):
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
    # print(gradient_vector.shape)
    return gradient_vector


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


# should this be a class that has acess to confgiruations?
# this function does it for each row indivdually!
def maxPSL_parallel(Jguess, configurations, analytical_gradient=False):
    B, N = configurations.shape
    # print(Jguess.shape, configurations.shape)
    # x = []
    # print(Jguess)
    if analytical_gradient is False:
        gradient_vector = None
    else:
        gradient_vector = PSL_row_gradient
    Jinferred = np.zeros_like(Jguess)
    # confgiruations =
    # iterator = zip(spins, guesses)
    const_args = (configurations, gradient_vector)
    spins = np.arange(0, N)
    guesses = np.array([guess for guess in Jguess])
    '''
    for spin, row_guess in zip(spins, guesses):
        loop_args = (spin, row_guess)
        inner_args = const_args + loop_args
        row_params = parallel_innerloop(inner_args)
        # print(row_params)
        Jinferred[spin] = row_params
    '''
    r = Parallel(n_jobs=-1)(
        delayed(parallel_innerloop)(const_args + (i, j))
        for i, j in zip(spins, guesses))
    # print(r)
    # print(len(r))
    r = np.array(r)
    # print(r.shape)
    # Jinf = np.array(r) would simplify this
    Jinferred = r
    '''
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
    '''
    # LL = np.sum(x)
    # print(LL)
    return Jinferred


def parallel_innerloop(inner_args):
    configurations, gradient_vector, spin, init_guess = inner_args
    #  = loop_params
    # init_guess, spin = iterator
    print('Row {}'.format(spin))
    # do the minimisation (giving each row)
    res = minimize(
        PSL_row,
        x0=init_guess,
        args=(spin, configurations),
        jac=gradient_vector,
        method='L-BFGS-B',
        # options={'disp': True, 'maxiter': 20}
    )
    # Jinferred[spin] = res.x
    return res.x


# I should write this so it can read anyhing, not just the MC stuff!
class PLMmax:
    def __init__(self, fname, dset_label):
        self.fname = fname
        self.data_directory = pathlib.Path(fname).parent.absolute()
        self.dset_label = dset_label
        # how to make this apply alwasy?
        # always call it configurations!
        # this needs some renamining!
        with Readhdf5_mc(fname) as f:
            # self.ds = f.read_many_datasets("configurations")
            self.ds = f.read_single_dataset(dset_label, "configurations")

    def writehdf5(self, inferred_model):
        # print(self.data_directory)
        model_file_path = os.path.join(self.data_directory, 'models.hdf5')
        # print(model_file_path)
        # x = os.path.isfile(model_file_path)
        # i can simplify this with append I think!
        # make it
        with h5py.File(model_file_path, 'a') as f:
            key_list = list(f['/'].keys())
            # empty check
            # print(key_list, len(key_list))
            if not key_list:
                grp_label = 'InferredModels:0'
            elif key_list[0] == 'TrueModels' and len(key_list) < 2:
                grp_label = 'InferredModels:0'
            else:
                grp_label = key_list[-1]
                split = grp_label.split(':')
                grp_label = split[0] + ':' + str(int(split[1]) + 1)
            print(grp_label)
            # this won't work for many things, it'll keep making
            # more and more groups! I should rething this!
            # or make the function read in ALL the GROUPS!
            group = f.create_group(grp_label)
            dataset = group.create_dataset(
                self.dset_label, data=inferred_model)
            dataset[()] = inferred_model
            print(list(f['/'].keys()))

    def infer(self, initial_guess_type):
        _, N = self.ds.shape
        if initial_guess_type == 'nMF':
            # i need to pick out speicfic datasets to do this!
            print(self.ds.shape)
            approx = analytical.Approximation(self.ds)
            initial_guess = approx.nMF()
        elif initial_guess_type == 'random':
            initial_guess = np.random.rand(N, N)
        else:
            initial_guess = None
            # return error!
        self.p0 = initial_guess

        s = timer()
        PLM_model = maxPSL_parallel(self.p0, self.ds, True)
        e = timer()
        print('\n\n')
        print(e - s)
        PLM_model = (PLM_model + PLM_model.T) / 2
        self.writehdf5(PLM_model)
        '''
        outpath = os.path.join(
            self.data_directory, 'PLM_inference_output.hdf5')
        with h5py.File(outpath, 'w') as f:
            # INCLUDE A THING THAT SAYS, IF TRUE MODEL KNOWN, DO BLAH
            # SO OPTION FOR TRUE MODELS!
            dataset = f.create_dataset(
                'model:' + self.dset_label, data=PLM_model)
            dataset[()] = PLM_model
            print(list(f.keys()))
        # save the inffered model
        '''
        return PLM_model
