import numpy as np
import matplotlib.pyplot as plt

import inference.analysis.new as analysis
from inference.io import Readhdf5_mc
from inference.core.pseudolikelihood import PLMmax
run_directory = 'PSL_ISING_TEST2'
fname = run_directory + '/mc_output.hdf5'
analysis.hdf5_plotObsAndFluc(fname)

pseudoLL_pipeline = PLMmax(fname)

print(vars(pseudoLL_pipeline))


# aha its to do with the order thigns get saved!! GOOD TO KNOW
# hurray I've fixed it!!
# now let's write a class that reads the thing!
