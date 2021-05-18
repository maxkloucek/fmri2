import glob
from os.path import join

from inference.core.pseudolikelihood import PLmax

run_base_directory = './datasetISING'
run_dirs = sorted(glob.glob(join(run_base_directory, '*')))
print(run_dirs)
for run_dir in run_dirs:
    print(run_dir)
    fname = join(run_dir, 'mc_output.hdf5')
    PLLM_pipeline = PLmax(fname)
    inf_model = PLLM_pipeline.infer('nMF')
