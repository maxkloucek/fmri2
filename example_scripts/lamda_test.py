import numpy as np
import matplotlib.pyplot as plt

import inference.analysis.planalysis as planalysis

from inference.io import Readhdf5_model
from inference.core.pseudolikelihood import PLmax

plt.style.use('~/Devel/styles/custom.mplstyle')

run_directory = 'PSL_SK_NewWeek'
fnameO = run_directory + '/mc_output.hdf5'
fnameM = run_directory + '/models.hdf5'
print('-----')


#   # how to do deal with labels?
# I want to be able to set the metadat of the infered stuff with e.g.
# lambda, and the time taken to do that inference!
'''
penalties = np.array([0.01])
PLLM_pipeline = PLmax(fnameO)
for p in penalties:
    inf_model = PLLM_pipeline.infer('nMF', l1_penalty=p)
'''
inference_labels = ['6', '8', '7']
for il in inference_labels:
    print(il)
    with Readhdf5_model(fnameM, show_metadata=False) as f:
        true_models, labels = f.read_multiple_models('TrueModels')
        inf_models, _ = f.read_multiple_models('InferredModels:' + il)
        # print(labels)
    # aha now I can loop thorugh the indices to do stuff!
    for tm, im in zip(true_models, inf_models):
        planalysis.overview(tm, im)
