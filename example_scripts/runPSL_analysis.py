from math import inf
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import diff

from sklearn.metrics import r2_score

import inference.analysis.new as analysis
import inference.tools as tools
from inference.io import Readhdf5, Readhdf5_mc, Readhdf5_model
from inference.core.pseudolikelihood import PLMmax
plt.style.use('~/Devel/styles/custom.mplstyle')


# let's make a reconstruction error function
# 2012 PRL: 1/(1/sqrt(N))<(Jinf - Jtrue)^2> * 0.5
# Nuygen  : sqrt(sum(Jinf - Jtru)^2 / sum(Jtru)^ 2)
# I could compare with the average value of a coupling?
# sure why not
def reconstruction_error(true_params, inf_params):
    # mean squared difference error
    N = true_params.size
    # print(N)
    diff = inf_params - true_params
    diff_sqr = diff ** 2
    mean_diff_sqr = np.mean(diff_sqr)
    msde = (mean_diff_sqr ** 0.5) / np.sqrt(N)
    return msde


def matrix_error(true_model, inferred_model):
    diff_sqr = (inferred_model - true_model) ** 2
    true_sum = np.sum(tools.triu_flat(true_model) ** 2)
    # print(true_sum)
    nguyen_error = (np.sum(tools.triu_flat(diff_sqr)) / true_sum) ** 0.5
    # print(nguyen_error)
    return diff_sqr  # / true_sum


def relative_error_wrt_mean(true_model, inferred_model):
    true_params = tools.triu_flat(true_model)
    # inf_params = tools.triu_flat(inf_model)
    # do two means?
    true_parameter_mean_sqr = np.mean(true_params ** 2)
    # print(true_parameter_mean_sqr)
    diff_sqr = (inferred_model - true_model) ** 2
    error_matrix = (diff_sqr / true_parameter_mean_sqr) ** 0.5
    return error_matrix


def mean_errors(true_model, inferred_model):
    # true_params = tools.triu_flat(true_model)
    # inf_params = tools.triu_flat(inf_model)
    # do two means?
    # true_parameter_mean_sqr = np.mean(true_params ** 2)
    # print(true_parameter_mean_sqr)
    diff_sqr = ((inferred_model - true_model) ** 2) ** 0.5
    abs_error = np.abs(inferred_model - true_model)
    # print(np.allclose(diff_sqr, abs_error))
    # error_matrix = (diff_sqr / true_parameter_mean_sqr) ** 0.5
    # return abs_error
    return diff_sqr


def overview(true_model, inf_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inf_model)

    # model_error = matrix_error(true_model, inf_model)
    # model_error = relative_error_wrt_mean(true_model, inf_model)
    model_error = mean_errors(true_model, inf_model)
    r2_val = r2_score(true_params, inf_params)
    # print(reconstruction_error(true_params, inf_params))

    max_value = np.max(inf_params)
    min_value = np.min(inf_params)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(true_model, vmin=min_value, vmax=max_value)
    ax[0].set(title='True Model')
    ax[1].imshow(inf_model, vmin=min_value, vmax=max_value)
    ax[1].set(title='Inferred Model')
    ax[2].imshow(model_error, vmin=min_value, vmax=max_value)
    ax[2].set(title='Error Matrix')
    ax[3].axline(
        (-1, -1), (1, 1), marker=',', color='k', transform=ax[3].transAxes)
    ax[3].plot(
        true_params, inf_params,
        linestyle='None',
        label=r'$R^{2}=$' + '{:.3f}'.format(r2_val))
    ax[3].set(xlabel='True', ylabel='Inferred', title='Reconstruction')
    plt.legend()
    plt.show()


# run_directory = 'PSL_ISING_TEST_CLEANAF'
run_directory = 'PSL_ISING_TEST2'
# fnameO = run_directory + '/mc_output.hdf5'
fnameM = run_directory + '/models.hdf5'
print('-----')
with Readhdf5_model(fnameM, show_metadata=True) as f:
    # labels = f.keys()

    # true_models = [f.read_single_dataset('TrueModels', 'T=1.00')]
    # inf_models = [f.read_single_dataset('InferredModels:0', 'T=1.00')]
    # print('----')
    true_models, labels = f.read_multiple_models('TrueModels')
    inf_models, _ = f.read_multiple_models('InferredModels:0')
    print(labels)
    # print('----')
    #  f.read_many_datasets('TrueModels')
    #  f.read_many_datasets('InferredModels:0')

for label, true_model, inf_model in zip(labels, true_models, inf_models):
    print(label)
    overview(true_model, inf_model)
'''
diff_matrix = inf_model - true_model
diff_params = tools.triu_flat(diff_matrix)
true_params = tools.triu_flat(true_model)
plt.hist(true_params, density=True, label='TrueDist')
plt.hist(diff_params, density=True, label='ErrorDist')
plt.show()
'''
'''
line_true = true_model[0, :]
line_inf = inf_model[0, :]

plt.plot(line_true)
plt.plot(line_inf)
plt.show()
'''