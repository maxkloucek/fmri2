import json
import numpy as np
# import matplotlib.pyplot as plt

import inference.core.io as io
import inference.core.utils as utils
# import inference.core.measures as m
import inference.core.pseudolikelihood as plmax
import inference.core.analytical as analytical
import inference.analysis.fitting as afit

import inference.sweep as s
from os.path import join
from pathlib import Path
from timeit import default_timer as timer


L = 3
N = 10  # L ** 2

rn = 'PSL_N{}_10^4datapoints'.format(N)
run_dir = join('.', rn)
print(run_dir)
Path(run_dir).mkdir(exist_ok=True)

cycles_eq = 1 * (10 ** 4)
cycles_prod = 1 * (10 ** 4)
reps = 1
cycle_dumpfreq = 10

pname = 'T'
# pvals = np.array([3])
# models =
# [aux.ising_interaction_matrix_2D_PBC2(L, T=val, h=0) for val in pvals]
# need to get a better way to put these values in here! Anyway, OK for now!
# i want to have histograms overlap!
pvals = np.array([1.0])
models = [
    utils.SK_interaction_matrix(N, T=val, h=0.0, jmean=1.0) for val in pvals]

model = models[0]

metadata = {
    'RunDirectory': run_dir,
    'SystemSize': L ** 2,
    'EqCycles': cycles_eq,
    'ProdCycles': cycles_prod,
    'CycleDumpFreq': cycle_dumpfreq,
    'Repetitions': reps,
    'SweepParameterName': pname,
    'SweepParameterValues': pvals.tolist()}

print('---- Generating True Model Data ----')
test_data_trajectory = s.simple_sim(
        run_dir, model, metadata, None,
        cycles_eq, cycles_prod, cycle_dumpfreq=10)


# si_data, sij_data, si_sj_data = m.correlations(test_data_trajectory)
# data_spin_matrix = aux.gen_spin_matrix(si_data, sij_data)
print('\n\n')

print('---- Analytical Calc ----')
# then jsut do a correaltions plot and im good right?
# I want to make this structure for the monte carlo as well!
# and as I've just shown I can make a class with the inner
# MC loop as its own function, which I think will help me
# clean up things a lot!
approx = analytical.Approximation(test_data_trajectory)
nMF_model = approx.nMF()

Jupper = nMF_model[np.triu_indices(N, k=1)]
# LL_F = plmax.pseudo_loglikelihoodF(Jupper, test_data_trajectory)
# s = timer()
# model_PSL = plmax.maxPSL(nMF_model, test_data_trajectory)
# e = timer()
# print(e - s)
# symmetric_model_PSL1 = (model_PSL + model_PSL.T) / 2
s = timer()
guess = nMF_model
# guess = np.zeros_like(nMF_model)
model_PSL = plmax.maxPSL_parallel(guess, test_data_trajectory, True)
e = timer()
print('\n\n')
print(e - s)
symmetric_model_PSL2 = (model_PSL + model_PSL.T) / 2

# print(np.allclose(symmetric_model_PSL1, symmetric_model_PSL2, atol=1e-06))
# symmetric
# print(symmetric_model_PSL1)
# print(symmetric_model_PSL2)
# plmax.PSL_row_gradient(model[0], 0, test_data_trajectory)
# print(model)
# print(model_PSL)
# print(symmetric_model_PSL)

io.save_npz(
    join(run_dir, 'mtrajectory'),
    test_model=model, model_traj=np.array([symmetric_model_PSL2]))

# io.save_npz(join(run_dir, 'T_mchi'), T=pvals, m=mags, chi=chis)
with open(join(run_dir, '_metadata.json'), 'w') as fout:
    json.dump(metadata, fout, indent=4)

saved_dir = run_dir + '/'
run_dataset = afit.Jdataset(saved_dir, save_figs=False)
run_dataset.compute_relative_error()
error = run_dataset.compute_errors(
    plot_trajectory=False, plot_matricies=True, plot_correlation=True)
run_dataset.compute_histogram(true_model=True)
# run_dataset.compute_histogram(true_model=False)
