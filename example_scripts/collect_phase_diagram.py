import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob

from os.path import join

import inference.analysis.planalysis as planalysis
from inference import tools
from inference.io import Readhdf5_model
# make a hdf5 file which collects everything!
# still want to save more metadata when I infer stuff!
plt.style.use('~/Devel/styles/custom.mplstyle')

run_base_directory = './datasetSK'
run_dirs = sorted(glob.glob(join(run_base_directory, '*')))
# error_phase_diagram:
with Readhdf5_model(run_dirs[0] + '/models.hdf5', show_metadata=False) as f:
    _, labels = f.read_multiple_models('TrueModels')

J0s = [float(run_dir.split('_')[1]) for run_dir in run_dirs]
temps = [float(label.split('=')[1]) for label in labels]

print(len(temps))
print(len(J0s))
error_phase_diagram = np.zeros((len(temps), len(J0s)))
cut = 20
for i, run_dir in enumerate(run_dirs[0:cut]):
    print(run_dir)
    fnameM = run_dir + '/models.hdf5'
    with Readhdf5_model(fnameM, show_metadata=False) as f:
        true_models, _ = f.read_multiple_models('TrueModels')
        inf_models, _ = f.read_multiple_models('InferredModels:0')

    errors = []
    for true_model, inf_model in zip(true_models, inf_models):
        true_params = tools.triu_flat(true_model)
        inf_params = tools.triu_flat(inf_model)
        # abs_error = np.mean(np.abs(inf_params - true_params))
        e = planalysis.reconstruction_error_nguyen(true_model, inf_model)
        errors.append(e)
    errors = np.array(errors)
    error_phase_diagram[:, i] = errors
    # planalysis.overview(true_model, inf_model)

# I need to adjust this so the centres fit
# today implement the metadata of the fitting!
lim = 0.3
limMin = np.min(error_phase_diagram)
'''
cm = mpl.cm.spring(np.linspace(1, 0, cut))
for c in range(0, cut):
    plt.plot(
        temps, error_phase_diagram[:, c],
        # linestyle='None',
        color=cm[c],
        label=r'$J_{0}=$' + '{:.4f}'.format(J0s[c]))
plt.xlabel(r'$T$')

plt.ylim(0, lim)
plt.legend(ncol=4)
plt.show()
'''
ext = np.array([np.min(J0s), np.max(J0s), np.min(temps), np.max(temps)])
J0_spacing = (J0s[1] - J0s[0]) / 2
temp_spacing = (temps[1] - temps[0]) / 2
adjustments = np.array([-J0_spacing, J0_spacing, -temp_spacing, temp_spacing])
ext = ext + adjustments

plt.imshow(
    error_phase_diagram, cmap='YlGnBu', vmax=lim, vmin=limMin, extent=ext)
plt.xlabel(r'$J_{0}$')
plt.ylabel(r'$T$')
plt.title(r'Error $(\epsilon)$')
plt.colorbar()
plt.show()
