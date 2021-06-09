import numpy as np
# import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path

import inference.core.utils as utils
from inference.sweep import MonteCarlo
# from inference.core.pseudolikelihood import PLmax

# this is to span the phase diagram!

# plt.style.use('~/Devel/styles/custom.mplstyle')
L = 10
N = L ** 2

run_base_directory = './datasetISINGshortSweep_N{}'.format(N)
Path(run_base_directory).mkdir(exist_ok=True)
# generate some datasets
# have the J0 be a thibng I can feed in!
# I could probably make this parallel too!!
# loool I never actually updated the hs! What a facepalm!
# makes sense now that nothing changed!
jmeans = np.linspace(0, 1.75, 20)
jmeans = np.linspace(-0.5, 0.5, 10)
temps = np.linspace(1.2, 5, 10)
for jmean in jmeans:
    run_label = 'h_' + '{:.4f}'.format(jmean)
    run_directory = join(run_base_directory, run_label)
    # ising_models = [
    #    utils.SK_interaction_matrix(
    #        N, T=val, h=0.0, jmean=jmean) for val in temps]
    ising_models = [
        utils.ising_interaction_matrix_2D_PBC2(L, T=val, h=jmean)
        for val in temps]
    ising_sim = MonteCarlo(run_directory)
    ising_sim.setHyperParameters(
        eq_cycles=1 * (10 ** 4),
        prod_cycles=1 * (10 ** 4)
        )
    ising_sim.describeSweepParameters('T', temps)
    ising_sim.setSweepModels(ising_models)
    ising_sim.run()
