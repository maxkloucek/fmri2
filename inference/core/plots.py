import numpy as np
import matplotlib.pyplot as plt


def configuration_2D(config, **kwargs):
    N = config.size
    L = int(np.sqrt(N))
    T = kwargs['T']
    config = config.reshape((L, L))
    plt.title('T = {}'.format(T))
    plt.imshow(config)
    plt.show()

# def mag_and_chi_sweep():
