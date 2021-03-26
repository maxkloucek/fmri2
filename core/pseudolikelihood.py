import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# coding up the stuff
# waximise the sum over the individaul things.
# " solve N independent gradient-descent problems in N variables"


def logistic_function(x):
    return 1 / (1 + np.exp(-x))


# x = np.linspace(-6, 6, 100)
# y = logistic_function(x)
# plt.plot(x, y)
# plt.show()



# I want do def f(x) where x is a vector of the parameters that minimize that
# I want to find
# args will be my data I suppose?
# maybe it means to update them all? I really don't know!
# work out porb and calc mean of all nd probs for all B
def row_log_likelihood(spin_index, configurations, parameter_matrix):
    probs = [
        conditional_probability(spin_index, config, parameter_matrix)
        for config in configurations]
    log_liklelihood = np.mean(np.log(probs))
    return log_liklelihood


def log_pseudo_liklelihoods(configurations, parameter_matrix):
    # N by N things
    B, N = configurations.shape
    print(B, N)
    likelihoods = [
        row_log_likelihood(spin_index, configurations, parameter_matrix)
        for spin_index in range(0, N)]
    likelihoods = np.array(likelihoods)
    print(likelihoods.shape)
    # for spin_index in range(0, N):
    # summing stuff
    # x is a 1D array length N for our minimsation here!
    return 0


# this could be used to do my MC flipping as well!!!!
# cause this is the local change!!!
# opportunity to clean up my code?
# sign convention?!?
# shall I call it row index? row index = spin index
def conditional_energy(spin_index, configuration, parameter_matrix):
    # -hisi -si sum(Jij sj) for j != i
    # paramter matrix (N,N) symmetric matrix, diagonal = vect(h)
    # plt.imshow(parameter_matrix)
    # plt.show()
    # print(configuration)
    # WRONG AND BROKEN!
    i = spin_index
    selected_spin = configuration[i]
    spin_products = np.copy(configuration) * selected_spin
    spin_products[i] = selected_spin
    parameter_row = parameter_matrix[i, :]
    E = - np.sum(parameter_row * spin_products)

    # print(parameter_matrix)
    # print(parameter_row)
    # print(selected_spin)
    # print(spin_products)  # ok I think this works!
    # print(E)
    return E


def conditional_probability(parameter_row, row_index, configuration):
    i = row_index
    selected_spin = configuration[i]
    spin_products = np.copy(configuration) * selected_spin
    spin_products[i] = selected_spin
    # -ves in here somewhere!!
    exponent = - 2 * np.sum(parameter_row * spin_products)
    probability = logistic_function(exponent)
    # print(probability)
    return probability


# configurations shape (B, N)
def sub_pseudo_likelihood(parameter_row, spin_index, configurations):
    # print(parameter_row, spin_index)
    B, N = configurations.shape
    log_probabilities = 0
    for sample in range(0, B):
        # config = configurations[sample, :]
        log_probabilities += np.log(conditional_probability(
            parameter_row, spin_index, configurations))
    # -ve of obejctive to minimize rather than amximise?
    return -(log_probabilities / B)


def maximise_row(spin_index, configurations):
    B, N = configurations.shape
    parameter_row0 = np.zeros(N)
    res = minimize(
        physics_f,
        x0=parameter_row0,
        args=(spin_index, configurations),
        method='Nelder-Mead',
        # options={'disp': True, 'maxiter': 20}
        )
    print(res.x)
    print(res.message)
    return res.x



def physics_f(parameters, spin_index, configurations):
    # si*hi + Jijsj for j!=i
    B, N = configurations.shape
    ln_probs = []
    for configuration in configurations:
        s_r = configuration[spin_index]
        s_vector = np.copy(configuration)
        s_vector[spin_index] = s_r
        exponent = -2 * np.sum(s_vector * parameters)
        # we minimize - this function in PRL!
        # maybe I had too many negatives kicking around!!
        ln_probs.append(np.log(1 + np.exp(exponent)))
    return np.mean(ln_probs)

# maybe I write a function for everything and try to minimize that
# here we go again!


def maximise_all(configurations):
    B, N = configurations.shape
    parameters_init = np.zeros((N, N))
    res = minimize(
        log_pseudo_likelihood,
        x0=parameters_init,
        args=(configurations),
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 20}
        )
    return res.x


# parameter_matrix has field on diagonal, couplings off diagonal
def log_pseudo_likelihood(parameter_matrix, configurations):
    B, N = configurations.shape
    row_log_likelihoods = []
    for spin_index in range(0, N):
        parameter_row = parameter_matrix[spin_index]
        row_log_likelihoods.append(
            sub_pseudo_likelihood(parameter_row, spin_index, configurations))
    return np.mean(row_log_likelihoods)
# whoops should have been mean not sum! (not sure this matters massively)?