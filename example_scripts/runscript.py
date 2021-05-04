import numpy as np
import matplotlib as pl
import matplotlib.pyplot as plt

# import src.core.aux as aux
# import src.core.measures as m
# import src.core.montecarlo as mc
# import setup
# setup.initialise(rn, y=4)
# import src.core.test_mc as test
# import src.core.io as io
import inference.core.test_mc as test
import inference.core.io as io
import intference.core.aux as aux
from os.path import join

L = 20
temps = np.linspace(2, 2.6, 3)
rn = 'run1'
# why dont I save an image of the equibrilated config?3 33333333333333333334444
# yaeh this sounds good!
cycles_eq = 1 * (10 ** 3)
cycles_prod = 1 * (10 ** 3)
reps = 1
# temps = np.array([2.2, 2.2, 2.2])
# temps = np.array([2.2])
# test_equil(temps, L, cycles)
# plt.legend()
# plt.show()
run_dir = join('.', rn)
print(run_dir)
mags, chis = test.test_zero_field(run_dir, temps, L, cycles_eq, cycles_prod, reps)
io.save_npz(join(run_dir, 'T_mchi'), T=temps, m=mags, chi=chis)
# with open('./runs/T_mchi.npz', 'wb') as fout:
#    np.savez(fout, T=temps, m=mags, chi=chis)
# critical slowing down is a problem (?)

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

# setup.finalise(rn)
