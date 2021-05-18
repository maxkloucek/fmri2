import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from inference import tools


def reconstruction_error_nguyen(true_model, inferred_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    numerator = np.sum((inf_params - true_params) ** 2)
    denominator = np.sum(true_params ** 2)
    return np.sqrt(numerator / denominator)


def overview(true_model, inf_model):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inf_model)

    model_error = np.abs(inf_model - true_model)
    r2_val = r2_score(true_params, inf_params)

    max_value = np.max(true_params)
    min_value = np.min(true_params)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(true_model, vmin=min_value, vmax=max_value)
    ax[0].set(title='True Model')
    ax[1].imshow(inf_model, vmin=min_value, vmax=max_value)
    ax[1].set(title='Inferred Model')
    ax[2].imshow(model_error, vmin=min_value, vmax=max_value)
    ax[2].set(title='Error Matrix')
    ax[3].axline(
        (-1, -1), (1, 1), marker=',', color='k', transform=ax[3].transAxes)
    ax[3].plot(
        true_params, inf_params,
        linestyle='None',
        label=r'$R^{2}=$' + '{:.3f}'.format(r2_val))
    ax[3].set(xlabel='True', ylabel='Inferred', title='Reconstruction')
    plt.legend()
    plt.show()
