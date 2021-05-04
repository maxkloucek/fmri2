import json
import numpy as np

import inference.core.io as io
import inference.core.aux as aux

from os.path import join
from pathlib import Path
from inference.sweep import parameter_sweep

# Ls = [3, 4, 5]
# Ls = [6, 8, 12, 16, 20]
# Ls = [25, 32, 48, 64]
Ls = [3]
for L in Ls:
    rn = 'N{}_1'.format(L ** 2)
    run_dir = join('.', rn)
    print(run_dir)
    Path(run_dir).mkdir(exist_ok=True)

    cycles_eq = 1 * (10 ** 5)
    cycles_prod = 1 * (10 ** 6)
    reps = 1
    cycle_dumpfreq = 10

    pname = 'T'
    pvals = np.linspace(1.8, 3.8, 50)

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
