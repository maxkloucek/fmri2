import json
import numpy as np

# import inference.core.io as io
import inference.core.aux as aux

from os.path import join
from pathlib import Path
from inference.sweep import parameter_sweep

Ls = [10]
for L in Ls:
    rn = 'N{}_1c'.format(L ** 2)
    run_dir = join('.', rn)
    print(run_dir)
    Path(run_dir).mkdir(exist_ok=True)

    cycles_eq = 1 * (10 ** 4)
    cycles_prod = 5 * (10 ** 4)
    reps = 1
    cycle_dumpfreq = 10

    pname = 'T'
    pvals = np.linspace(0.5, 5, 10)
    # pvals = np.array([2.26, 5])
    models = [
        aux.ising_interaction_matrix_2D_PBC2(L, T=val) for val in pvals]

    metadata = {
        'RunDirectory': run_dir,
        'SystemSize': L ** 2,
        'EqCycles': cycles_eq,
        'ProdCycles': cycles_prod,
        'CycleDumpFreq': cycle_dumpfreq,
        'Repetitions': reps,
        'SweepParameterName': pname,
        'SweepParameterValues': pvals.tolist()}

    parameter_sweep(
        run_dir, models, metadata,
        cycles_eq, cycles_prod, reps, cycle_dumpfreq)
    # io.save_npz(join(run_dir, 'T_mchi'), T=pvals, m=mags, chi=chis)
    with open(join(run_dir, '_metadata.json'), 'w') as fout:
        json.dump(metadata, fout, indent=4)

# models = [aux.ising_interaction_matrix_2D_PBC(L, 0, 1 / T) for T in pvals]
