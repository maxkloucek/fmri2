import numpy as np
import matplotlib.pyplot as plt

import inference.core.aux as aux
from inference.sweep import MonteCarlo
from inference.core.pseudolikelihood import PLMmax


plt.style.use('~/Devel/styles/custom.mplstyle')
run_directory = 'PSL_ISING_TEST2'

# generate some datasets
L = 5
N = L ** 2

# temps = np.linspace(0.5, 15, 20)
temps = [0.5, 1, 1.5]  # lists get converted to np arrays
temps = [0.8, 1, 1.2]  # lists get converted to np arrays
temps = np.linspace(0.8, 1.4, 5)
ising_models = [
    aux.ising_interaction_matrix_2D_PBC2(L, T=val) for val in temps]

ising_models = [
    aux.SK_interaction_matrix(N, T=val, h=0.0, jmean=0.0) for val in temps]

ising_sim = MonteCarlo(run_directory)
ising_sim.setHyperParameters(
    eq_cycles=1 * (10 ** 4),
    prod_cycles=5 * (10 ** 4)
    )
ising_sim.describeSweepParameters('T', temps)
ising_sim.setSweepModels(ising_models)
ising_sim.run()
# I WANT TO COMBINE SWEEP PARAMETERS AND MODELS?

fname = run_directory + '/mc_output.hdf5'
print(fname)
# analysis.hdf5_plotObsAndFluc(fname)
PLLM_pipeline = PLMmax(fname, dset_label='qwerbl')
# PLLM_pipeline.check_group_contents()
# PLLM_pipeline.setup_groups()
inf_model = PLLM_pipeline.infer('nMF')
