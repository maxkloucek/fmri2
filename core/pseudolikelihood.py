import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

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
    plt.imshow(J)
    plt.show()
    # note: -log 1/P = log P
    f = np.mean(np.log(
        1+np.exp(
            -2*beta*(spin@J)*spin)
    ),
        axis=0
    )
    print(f)
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


def PSL_row_gradient():
    return 0


# this function does it for each row indivdually!
def maxPSL(Jguess, configurations):
    B, N = configurations.shape
    print(Jguess.shape, configurations.shape)
    # x = []
    print(Jguess)
    Jinferred = np.zeros_like(Jguess)
    for spin in range(0, N):
        print('Row {} of {}'.format(spin, N))
        # do the minimisation (giving each row)
        res = minimize(
            PSL_row,
            x0=Jguess[spin],
            args=(spin, configurations),
            method='L-BFGS-B'
            # options={'disp': True, 'maxiter': 20}
        )
        Jinferred[spin] = res.x
        # x.append(PSL_row(res.x, spin, configurations))
    # LL = np.sum(x)
    # print(LL)
    return Jinferred
