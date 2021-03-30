import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from joblib import Parallel, delayed

# coding up the stuff
# waximise the sum over the individaul things.
# " solve N independent gradient-descent problems in N variables"


def logistic_function(x):
    return 1 / (1 + np.exp(-x))


def pseudo_loglikelihoodF(Jtriu, spin=None): #add lamba parameter* norm of J sum of abs values (sum aboslute values and multiply by lambda)
    N = spin.shape[1]
    B = spin.shape[0]
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
    res = Parallel(n_jobs=1, verbose=10)(delayed(np.sqrt)(i) for i in range(10))
    print(res)
    B, N = configurations.shape
    print(Jguess.shape, configurations.shape)
    # x = []
    print(Jguess)
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
