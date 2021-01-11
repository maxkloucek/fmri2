import numpy as np


def ising_interaction_matrix_2D_PBC(L, h, jval):
    N = L ** 2
    J = np.zeros((N, N))
    for i in range(0, N):
        if i % L == 0:
            J[i, i+L-1] = jval
        else:
            J[i, i-1] = jval

        if (i+1) % L == 0:
            J[i, i-L+1] = jval
        else:
            J[i, i+1] = jval

        if i < L:
            J[i, i + (N-L)] = jval
        else:
            J[i, i-L] = jval

        if i >= (N-L):
            J[i, i-(N-L)] = jval
        else:
            J[i, i+L] = jval
    np.fill_diagonal(J, h)
    return J


# return 1D object containing list of initial Ising Spins
# (i.e. +/- 1 only -> binary)
def initialise_ising_config(N, option):
    if option == -1:
        config = -np.ones(N)
    elif option == 0:
        config = np.random.randint(2, size=N)
        config[config == 0] = -1
    elif option == 1:
        config = np.ones(N)
    else:
        print('Invalid initialisation choice made')
        return 1
    return config
