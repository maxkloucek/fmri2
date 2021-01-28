import numpy as np
import matplotlib as pl
import matplotlib.pyplot as plt


with open('T_mchi.npz', 'rb') as fin:
    traj = np.load(fin)
    temps = traj['T']
    mags = traj['m']
    chis = traj['chi']


cols = pl.cm.viridis(np.linspace(0, 1, 2))
fig, ax = plt.subplots()
twinned_ax = ax.twinx()
l1 = ax.errorbar(
    temps, mags[0, :], mags[1, :],
    marker='o', color=cols[0], label=r'$|m|$')
l2 = twinned_ax.errorbar(
    temps, chis[0, :], chis[1, :],
    marker='v', color=cols[1], label=r'$\chi$')
ax.set(xlabel='T', ylabel=r'$|m|$')
twinned_ax.set(ylabel=r'$\chi$')

lines = [l1, l2]
ax.legend(lines, [line.get_label() for line in lines])
plt.show()
