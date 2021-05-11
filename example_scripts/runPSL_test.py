import numpy as np
import matplotlib.pyplot as plt

import inference.core.aux as aux
from inference.sweep import MonteCarlo
from inference.core.pseudolikelihood import PLMmax

plt.style.use('~/Devel/styles/custom.mplstyle')


# generate some datasets
L = 5
N = L ** 2
run_directory = 'PSL_ISING_TEST2'
temps = np.linspace(0.5, 15, 20)
temps = [2]  # lists get converted to np arrays
ising_models = [
    aux.ising_interaction_matrix_2D_PBC2(L, T=val) for val in temps]

# ising_models = [
#    aux.SK_interaction_matrix(N, T=val, h=0.0, jmean=0.0) for val in temps]

ising_sim = MonteCarlo(run_directory)
ising_sim.setHyperParameters(
    eq_cycles=1 * (10 ** 4),
    prod_cycles=1 * (10 ** 4)
    )
ising_sim.describeSweepParameters('T', temps)
ising_sim.setSweepModels(ising_models)
ising_sim.run()
# I WANT TO COMBINE SWEEP PARAMETERS AND MODELS?


# models = [aux.ising_interaction_matrix_2D_PBC(L, 0, 1 / T) for T in pvals]
# give it a function to generate?
# mc_sweep(mc_metadata, models)

# fname = run_directory + '/mc_output.hdf5'
fname = run_directory + '/mc_output.hdf5'
print(fname)
# analysis.hdf5_plotObsAndFluc(fname)
PLLM_pipeline = PLMmax(fname, 'T=2.00')
inf_model = PLLM_pipeline.infer('random')
print(PLLM_pipeline.ds.shape)
print(PLLM_pipeline.p0.shape)
print(inf_model.shape)

min_val = np.min(ising_models[0])
max_val = np.max(ising_models[0])
diff = ising_models[0] - inf_model

fig, ax = plt.subplots(4, 1)
ax = ax.ravel()
ax[0].imshow(
    ising_models[0], vmin=min_val, vmax=max_val)
ax[1].imshow(
    PLLM_pipeline.p0, vmin=min_val, vmax=max_val)
ax[2].imshow(
    inf_model, vmin=min_val, vmax=max_val)
im3 = ax[3].imshow(
    diff, vmin=min_val, vmax=max_val)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im3, cax=cbar_ax)
plt.show()
