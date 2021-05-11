import numpy as np
import matplotlib.pyplot as plt
import inference.io as iohdf5


def hdf5_plotObsAndFluc(data_fname):
    # Tc = 2.269J/kb!
    # think I want to call it recalc!
    lM = r'<$ |M| / N $>'
    lChi = r'$\chi ^{*} / N$'
    lE = r'<$E / N$>'
    lCv = r'$C_ {v} / N$'

    labels = [lM, lChi, lE, lCv]
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    for a, l in zip(ax.ravel(), labels):
        a.set(xlabel='T', ylabel=l)
        a.axvline(2.269, marker=',', c='k')
    with iohdf5.Readhdf5_mc(data_fname, prod_only=True) as fin:
        metadata = fin.get_metadata()
        configs_all_T = fin.read_many_datasets("configurations")
        energy_all_T = fin.read_many_datasets("energy")

    sweep_paramters = np.array(metadata["SweepParameterValues"])
    # print(sweep_paramters.shape)
    # I THINK I SAVE E AS E/T * NEED TO TIMES IT BY T TO MAKE SURE
    # THAT NOTHING BAD IS HAPPENING!
    # YP SAVING E/T ATM KEEP THIS IN MIND!!!!
    output_aray = np.empty((4, sweep_paramters.size))

    for c, sweep_param in enumerate(sweep_paramters):
        print(c, sweep_param)
        # Ts = np.array(md["SweepParameterValues"])
        N = metadata["SystemSize"]
        # print(configs_all_T[c].shape)
        M_traj = np.sum(configs_all_T[c], axis=1)
        Mabs_traj = abs(M_traj)
        Mabs_mean = np.mean(Mabs_traj) / N
        Mabs_fluct = np.var(Mabs_traj) / sweep_param

        E_traj = energy_all_T[c] * sweep_param
        E_mean = np.mean(E_traj) / N
        E_fluct = np.var(E_traj) / (sweep_param ** 2)
        output_aray[0, c] = Mabs_mean
        output_aray[1, c] = Mabs_fluct
        output_aray[2, c] = E_mean
        output_aray[3, c] = E_fluct

    ax[0, 0].plot(
        sweep_paramters, output_aray[0, :], '.', label='N = {}'.format(N))
    ax[0, 1].plot(sweep_paramters, output_aray[1, :] / N, '.')
    ax[1, 0].plot(sweep_paramters, output_aray[2, :], '.')
    ax[1, 1].plot(sweep_paramters, output_aray[3, :] / N, '.')

    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()
