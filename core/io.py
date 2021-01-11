import numpy as np
import matplotlib.pyplot as plt
import os

from os.path import join
# from .utils import RUN_DIR
RUN_DIR = '/Users/mk14423/Desktop/code/0-fmri/fmri2/run1'

# I want these to have the full path already!
# append an identifier, e.g. .npz for data, and .config.png
# for configurations!

def save_npz(fname, **kwargs):
    # path_out = os.path.join(RUN_DIR, fname)
    path_out = fname + '.npz'
    print(path_out)
    # print(path_out)
    with open(path_out, 'wb') as fout:
        np.savez(fout, **kwargs)


def save_2Dconfig_image(fname, config, **kwargs):
    N = config.size
    L = int(np.sqrt(N))
    T = kwargs['T']

    # path_out = os.path.join(RUN_DIR, fname)
    path_out = fname + 'config.png'

    config = config.reshape((L, L))
    plt.title('{:.2f}'.format(T))
    plt.imshow(config)
    plt.savefig(path_out, dpi=600)
    plt.close()
    # plt.show()
