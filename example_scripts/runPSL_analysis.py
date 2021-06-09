import numpy as np
import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error

import inference.analysis.new as analysis
import inference.analysis.planalysis as planalysis
import inference.tools as tools
from inference.io import Readhdf5_model, Readhdf5_mc # Readhdf5
# from inference.core.pseudolikelihood import PLMmax
plt.style.use('~/Devel/styles/custom.mplstyle')


# let's make a reconstruction error function
# 2012 PRL: 1/(1/sqrt(N))<(Jinf - Jtrue)^2> * 0.5
# Nuygen  : sqrt(sum(Jinf - Jtru)^2 / sum(Jtru)^ 2)
# I could compare with the average value of a coupling?
# https://stats.stackexchange.com
# /questions/194278/meaning-of-reconstruction-error-in-pca-and-lda#
# :~:text=The%20general%20definition%20of%20the,subspace%20(its%20'estimate').
'''
def reconstruction_error_nguyen(true_model, inferred_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    numerator = np.sum((inf_params - true_params) ** 2)
    denominator = np.sum(true_params ** 2)
    return np.sqrt(numerator / denominator)
'''


def reconstruction_error_PRL(true_model, inferred_model):
    N, _ = true_model.shape
    print(N)
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    prefactor = 1 / (1 / np.sqrt(N))
    average_diff_sqared = np.mean((inf_params - true_params) ** 2)
    return prefactor * np.sqrt(average_diff_sqared)


# run_directory = 'PSL_ISING_TEST_CLEANAF'
# run_directory = 'PSL_ISING_TIGHTBANGER'
# L1 helps if sparse, but what if setting falsely to
# 0, when should be small not 0!!?
# so maybe no lambda is best in this setting?
# run_directory = 'datasetSK/J0_0.0000'
# run_directory = 'datasetISINGsmall_N25/h_0.0000'
run_directory = 'datasetISINGshortSweep_N100/h_0.0556'
fnameO = run_directory + '/mc_output.hdf5'
fnameM = run_directory + '/models.hdf5'

print('-----')  # this does the model error stuff!
with Readhdf5_model(fnameM, show_metadata=False) as f:
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

for label, true_model, inf_model in zip(labels, true_models, inf_models):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inf_model)
    # abs_error = np.mean(np.abs(inf_params - true_params))
    e = planalysis.reconstruction_error_nguyen(true_model, inf_model)
    errors.append(e)
    e2s.append(reconstruction_error_PRL(true_model, inf_model))
    print(label)
    T = float(label.split('=')[1])
    temps.append(T)
    # planalysis.overview(true_model, inf_model)

M, chi = analysis.hdf5_Mchi(run_directory + '/mc_output.hdf5')
with Readhdf5_mc(fnameO, False) as f:
    trajs = f.read_many_datasets('configurations')
    md = f.get_metadata()
    temps = md['SweepParameterValues']

ms = []
qs = []
# calculate q = 1/N * sum_i(<si>^2)
for trajectory in trajs:
    si_averages = np.mean(trajectory, axis=0)
    m = np.mean(si_averages)
    q = np.mean(si_averages ** 2)
    qs.append(q)
    ms.append(m)
# print(trajectory.shape, si_averages.shape)
# print(q)
min_error = np.min(errors)
chi_max = np.max(chi) * 1.02
fig, ax = plt.subplots()
ax.plot(temps, errors, label=r'$\epsilon$')
ax.plot(temps, M, label=r'$\|m\|$')
# plt.plot(temps, ms, label=r'$m$', color='grey')
ax.plot(temps, chi, label=r'$\frac{\chi}{N}$')
# plt.plot(temps, qs, label=r'$q$')
ax.axhline(min_error, marker=',', c='k')
ax.set_ylim(0, chi_max)
ax.set_yscale('linear')
plt.legend()
plt.show()
# plt.plot(temps, e2s, label='rando')

'''
diff_matrix = inf_model - true_model
diff_params = tools.triu_flat(diff_matrix)
true_params = tools.triu_flat(true_model)
plt.hist(true_params, density=True, label='TrueDist')
plt.hist(diff_params, density=True, label='ErrorDist')
plt.show()

line_true = true_model[0, :]
line_inf = inf_model[0, :]

plt.plot(line_true)
plt.plot(line_inf)
plt.show()
'''