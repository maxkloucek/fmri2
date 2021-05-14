import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error

import inference.analysis.new as analysis
import inference.tools as tools
from inference.io import Readhdf5_model  # Readhdf5, Readhdf5_mc
# from inference.core.pseudolikelihood import PLMmax
plt.style.use('~/Devel/styles/custom.mplstyle')


# let's make a reconstruction error function
# 2012 PRL: 1/(1/sqrt(N))<(Jinf - Jtrue)^2> * 0.5
# Nuygen  : sqrt(sum(Jinf - Jtru)^2 / sum(Jtru)^ 2)
# I could compare with the average value of a coupling?
# https://stats.stackexchange.com
# /questions/194278/meaning-of-reconstruction-error-in-pca-and-lda#
# :~:text=The%20general%20definition%20of%20the,subspace%20(its%20'estimate').
def reconstruction_error_nguyen(true_model, inferred_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    numerator = np.sum((inf_params - true_params) ** 2)
    denominator = np.sum(true_params ** 2)
    return np.sqrt(numerator / denominator)


def reconstruction_error_PRL(true_model, inferred_model):
    N, _ = true_model.shape
    print(N)
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    prefactor = 1 / (1 / np.sqrt(N))
    average_diff_sqared = np.mean((inf_params - true_params) ** 2)
    return prefactor * np.sqrt(average_diff_sqared)


def matrix_error(true_model, inferred_model):
    diff_sqr = (inferred_model - true_model) ** 2
    return diff_sqr


def mean_errors(true_model, inferred_model):
    diff_sqr = ((inferred_model - true_model) ** 2) ** 0.5
    # abs_error = np.abs(inferred_model - true_model)
    return diff_sqr


def overview(true_model, inf_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inf_model)

    # model_error = matrix_error(true_model, inf_model)
    # model_error = relative_error_wrt_mean(true_model, inf_model)
    model_error = mean_errors(true_model, inf_model)
    r2_val = r2_score(true_params, inf_params)
    # print(reconstruction_error(true_params, inf_params))

    max_value = np.max(true_params)
    min_value = np.min(true_params)

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
run_directory = 'PSL_ISING_TIGHTBANGER'
# run_directory = 'PSL_ISING_TEST4'
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

errors = []
e2s = []
temps = []
# true_models = [true_models[-1]]
# inf_models = [inf_models[-1]]
for label, true_model, inf_model in zip(labels, true_models, inf_models):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inf_model)
    # abs_error = np.mean(np.abs(inf_params - true_params))
    e = reconstruction_error_nguyen(true_model, inf_model)
    errors.append(e)
    # they are the same!
    # rmse = np.sqrt(mean_squared_error(true_params, inf_params))

    # RMSE normalised by mean:
    # nrmse = rmse/np.sqrt(np.mean(true_params**2))
    # r2_score(true_params, inf_params)
    # e2s.append(nrmse)
    e2s.append(reconstruction_error_PRL(true_model, inf_model))
    print(label)
    T = float(label.split('=')[1])
    temps.append(T)
    # overview(true_model, inf_model)
    # let's store the tmperatures in the metdata of Truemodesl

M, chi = analysis.hdf5_Mchi(run_directory + '/mc_output.hdf5')

plt.plot(temps, errors, label=r'$\epsilon$')
plt.plot(temps, M, label=r'$\|m\|$')
plt.plot(temps, chi, label=r'$\frac{\chi}{N}$')
# plt.axvline(2.269, marker=',', c='k')
plt.ylim(0, 1.5)
plt.legend()
plt.show()
# plt.plot(temps, e2s, label='rando')

# plt.show()
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