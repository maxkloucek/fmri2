import numpy as np
import matplotlib.pyplot as plt
import inference.analysis.trajectory as trajectory
import inference.io as io
plt.style.use('~/Devel/styles/custom.mplstyle')

'''
with io.Readhdf5_MC('full_dataset_mc.hdf5', prod_only=True) as fin:
    # configs = hdf5_file.read_single_dataset('T=2.26', 'energy')
    configs_all_T = fin.read_many_datasets("configurations")
    energy_all_T = fin.read_many_datasets("energy")

fig, ax = plt.subplots(2, 1)
ax = ax.ravel()
for configs, energies in zip(configs_all_T, energy_all_T):
    mag = np.mean(configs, axis=1)
    ax[0].plot(mag)
    ax[1].plot(energies / 100)
plt.show()
'''

# run_dirs = ['./N400_2/']
run_dirs = ['./N100_ncompress/']
# run_dirs = ['./N25_compress/']
fname = run_dirs[0] + 'full_dataset_mc.hdf5'
print(fname)
trajectory.hdf5_plotObsAndFluc(fname)
# should I put the two trajectories together?
# analysis.plotAutoCorr(run_dirs)
# trajectory.magchi(run_dirs)
# trajectoryE(rd, statepoints, Ts)
# traj(rd, statepoints, Ts)
# hist(rd, statepoints, Ts)
# plt 4, maybe I hsould have a plt single and a plt 2 function?
# not sure how best to standardise this!
# ----- Shape is B,N ---- #
# I could write a check metadata bash script!
# that might be useful!!
