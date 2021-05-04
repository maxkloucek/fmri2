import json
import numpy as np
import matplotlib.pyplot as plt

import inference.core.io as io
import inference.core.aux as aux
import inference.core.measures as m

import inference.sweep as s
import inference.core.analytical as analytical

from os.path import join
from pathlib import Path
from inference.sweep import parameter_sweep
from inference.sweep import IsingInference
from inference.core import montecarlo as mc
# th
# from inference.sweep import simple_sim
# from inference.sweep import gradient_descent
# Ls = [3, 4, 5]
# Ls = [6, 8, 12, 16, 20]
# Ls = [25, 32, 48, 64]
L = 3
N = L ** 2
# N = 75
# oops! this is a typo for N, so N accidentally ended up as 27!
# well, glad to know this works at least for bigger system
# in a somewhat reaasonable time haha!
# GDtest_N9_IsingNeighboursVar_weakinit_T3
# GDtest_N9
# maybe I have to set LR as a function of N to get convergence?
rn = 'RESTART_GDtest_N{}_SK'.format(L ** 2)
run_dir = join('.', rn)
print(run_dir)
Path(run_dir).mkdir(exist_ok=True)

cycles_eq = 1 * (10 ** 4)
cycles_prod = 1 * (10 ** 6)
reps = 1
cycle_dumpfreq = 10

pname = 'T'
# pvals = np.array([3])
# models = [aux.ising_interaction_matrix_2D_PBC2(L, T=val) for val in pvals]
# only works in itermediate area I'm pretty sure
pvals = np.array([0.9])
models = [aux.SK_interaction_matrix(N, T=val, h=0.1, jmean=0.8) for val in pvals]

model = models[0]
# plt.imshow(model)
# plt.colorbar()
# plt.show()
# save a name for the model!?
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
        1 * (10 ** 4), 1 * (10 ** 6), cycle_dumpfreq=10)

si_data, sij_data, si_sj_data = m.correlations(test_data_trajectory)
data_spin_matrix = aux.gen_spin_matrix(si_data, sij_data)
print('\n\n')
print('---- Running Gradient Descent ----')
'''
model_trajectory = s.gradient_descent(
    run_dir, 0.1 * np.ones(data_spin_matrix.shape), data_spin_matrix, metadata,
    GD_steps=600, MCCs_eq=cycles_eq, MCCs_prod=cycles_prod,
    cycle_dumpfreq=10)
'''
'''
model_trajectory = s.gradient_descent_anneal(
    run_dir, 0.1 * np.ones(data_spin_matrix.shape), data_spin_matrix, metadata,
    GD_steps=2500, MCCs_init_eq=cycles_eq, MCCs_anneal_step=cycles_prod,
    cycle_dumpfreq=10)
'''
infpipe = IsingInference(data_spin_matrix, run_dir)
approx = analytical.Approximation(test_data_trajectory)

init_model = approx.nMF()
# init_model = 0.1 * np.ones(data_spin_matrix.shape)
fig, ax = plt.subplots(1, 2)
ax.ravel()
ax[0].imshow(model)
ax[1].imshow(init_model)
plt.show()
# yep think I need alpha to be a function of N for some reason?!
model_trajectory, diffs = infpipe.fit_MonteCarlo(
    init_model, 0.1, 200, cycles_eq, cycles_prod)

plt.plot(diffs)
plt.show()
# I can save the model, and reinitialise it with the last one! that might be useful!!
# save model and trajecotry in one .npz file?
# tehshold fixed or adaptive?
# who knows what I'm doing here!
io.save_npz(
    join(run_dir, 'mtrajectory'),
    test_model=model, model_traj=model_trajectory)

# io.save_npz(join(run_dir, 'T_mchi'), T=pvals, m=mags, chi=chis)
with open(join(run_dir, '_metadata.json'), 'w') as fout:
    json.dump(metadata, fout, indent=4)
