import numpy as np
import matplotlib.pyplot as plt

# from os.path import join
# from .utils import RUN_DIR

# I want these to have the full path already!
# append an identifier, e.g. .npz for data, and .config.png
# for configurations!
# this will become obsolete soon!!


def save_npz(fname, **kwargs):
    path_out = fname + '.npz'
    with open(path_out, 'wb') as fout:
        np.savez(fout, **kwargs)


def save_2Dconfig_image(fname, config, label, labelval):
    N = config.size
    L = int(np.sqrt(N))
    path_out = fname + 'config.png'

    config = config.reshape((L, L))
    plt.title('{} = {:.2f}'.format(label, labelval))
    plt.imshow(config)
    plt.savefig(path_out, dpi=600)
    plt.close()
    # plt.show()


def check_output_file_exists():
    return 0
