import numpy as np
# import matplotlib as pl
# import matplotlib.pyplot as plt
from os.path import join

from . import aux
# from . import plots
from . import measures as m
from . import montecarlo as mc
from . import io

# I need to have some eq period, followed by a production period!!
# cause atm this doesnt make all that much sense
# when I'm pretty certain that's wrong!!
# maybe I should return the full data, and save it and then do averaging?
def test_zero_field(
        run_directory, temps,
        L=20, MCCs_eq=100000, MCCs_prod=100000, reps=1, dump_freq=10):
    N = L ** 2
    mags = np.zeros(reps)
    chis = np.zeros(reps)
    mags_averaged = np.zeros((2, temps.size))
    chis_averaged = np.zeros((2, temps.size))
    for c, T in enumerate(temps):
        print(c, T)
        model = aux.ising_interaction_matrix_2D_PBC(L, 0, 1 / T)
        for rep in range(0, reps):
            initial_config = aux.initialise_ising_config(N, 0)

            eq_traj = mc.simulate(model, initial_config, MCCs_eq, dump_freq)
            eq_config = eq_traj[-1]
            fname = join(run_directory, str(c) + 'eq')
            io.save_2Dconfig_image(fname, eq_config, T=T)

            prod_traj = mc.simulate(model, eq_config, MCCs_prod, dump_freq)
            fname = join(run_directory, str(c) + 'fin')
            io.save_2Dconfig_image(fname, prod_traj[-1], T=T)

            fname = join(run_directory, 'c{}r{}trajectory'.format(c, rep))
            print(fname)
            io.save_npz(fname, eq_traj=eq_traj, prod_traj=prod_traj)

            si, sij, si_sj = m.correlations2(prod_traj)
            cij = sij - si_sj
            mag = np.mean(si)
            chi = np.sum(cij)/(N*T)
            mags[rep] = abs(mag)  # this is acutally the absolute magnitude!
            chis[rep] = chi
        print(mags)
        print(chis)
        mags_averaged[0, c] = np.mean(mags)
        mags_averaged[1, c] = np.std(mags)
        chis_averaged[0, c] = np.mean(chis)
        chis_averaged[1, c] = np.std(chis)
    return mags_averaged, chis_averaged

# do I want to average si and so on???? maybe???
# now this is the key question...
# or maybe I want to run a few times from intialisation, but dont average each time?!


def test_equil(temps, L=20, MCCs=100000, dump_freq=10):
    N = L ** 2
    for T in temps:
        model = aux.ising_interaction_matrix_2D_PBC(L, 0, 1 / T)
        initial_config = aux.initialise_ising_config(N, 0)
        configs = mc.simulate(model, initial_config, MCCs, dump_freq)
        # si, sij, si_sj = m.correlations(configs)
        # cij = sij - si_sj
        # mag = np.mean(si)
        # chi = np.sum(cij)/(N*T)

        #what can I trust?
        si2, sij2, si_sj2 = m.correlations2(configs)
        # cij2 = sij2 - si_sj2
        # print(np.sum(si_sj), np.sum(si_sj2))
        # print(np.allclose(si, si2), np.allclose(sij, sij2), np.allclose(si_sj, si_sj2))
        # mag_traj = m.trajectory(configs, m.magnetisation)
        # plt.plot(mag_traj, label='{:.3f}, {:.3f}'.format(mag, chi))
        # plt.plot(energy_traj, label='energy density')
    return 0

'''
L = 20
temps = np.linspace(2, 2.6, 10)
# why dont I save an image of the equibrilated config?
# yaeh this sounds good!
cycles_eq = 1 * (10 ** 3)
cycles_prod = 1 * (10 ** 3)
reps = 1
# temps = np.array([2.2, 2.2, 2.2])
# temps = np.array([2.2])
# test_equil(temps, L, cycles)
# plt.legend()
# plt.show()

mags, chis = test_zero_field(temps, L, cycles_eq, cycles_prod, reps)

# critical slowing down is a problem (?)

cols = pl.cm.viridis(np.linspace(0, 1, 2))
fig, ax = plt.subplots()
twinned_ax = ax.twinx()
# l1, = ax.plot(temps, mags[0, :], marker='o', color=cols[0], label=r'$|m|$')
# l2, = twinned_ax.plot(temps, chis[0, :], marker='v', color=cols[1], label=r'$\chi$')
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
'''
# plt.savefig(run_directory + 'figPTtraj.png', dpi=600)
# plt.close()
'''
N = L ** 2
T = 2.26
print(N, T)
model = aux.ising_interaction_matrix_2D_PBC(L, 0, 1 / T)
initial_config = aux.initialise_ising_config(N, 0)


MCCs_eq = 1 * (10 ** 5)
print(MCCs_eq)
dump_freq = 10  # save data every 10 MCCs
configs = mc.simulate(model, initial_config, MCCs_eq, dump_freq)
print('Finished sim')
si, sij, si_sj = m.correlations(configs)
print('Finished measure')

'''

# plots.configuration(configs[-1])
# mag_traj = m.trajectory(configs, m.magnetisation)
# energy_traj = m.trajectory(configs, m.energy, J=model) / N
# plt.plot(mag_traj, label='mag')
# plt.plot(energy_traj, label='energy density')
# plt.legend(title='T = {}'.format(T))
# plt.show()
# si, sij, si_sj = m.correlations(configs)  # this cacls for the whole dataset!
# print(si.shape)
# si_abs = abs(si)

