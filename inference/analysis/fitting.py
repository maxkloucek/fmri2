import numpy as np
import matplotlib.pyplot as plt

from .trajectory import get_metadata
from .animate import animate2Dtrajectory
from ..core import measures as m

from scipy.stats import pearsonr
# should I have a class that can do a bunch of things?
# I think this is a better way to srttuctrue it!!
# don't fuck with this right now!!!
# figure out how exactly you want to save your data!
# have a save/ plt thing!!


class Jdataset:
    def __init__(self, run_dir, save_figs=False):
        fname = run_dir + 'mtrajectory.npz'
        with open(fname, 'rb') as fin:
            traj = np.load(fin)
            self.true_model = traj['test_model']
            self.model_trajectory = traj['model_traj']
        self.metadata = get_metadata(run_dir)
        self.run_dir = run_dir
        self.save_figs = save_figs

    def modify_sample_size(self, start_index, subsample_size):
        self.true_model = self.true_model[
            start_index: subsample_size, start_index: subsample_size]
        self.model_trajectory = self.model_trajectory[
            :, start_index: subsample_size, start_index: subsample_size]

    # returns split and flatrned hs and Js for further analysis!
    def split_diagonal(self, model):
        N, _ = model.shape
        hs = np.diagonal(model)
        Js = model[np.triu_indices(N, k=1)]
        return hs, Js

    def reform_split_trajectory(self):
        h_traj = []
        J_traj = []
        for model in self.model_trajectory:
            hs, Js = self.split_diagonal(model)
            h_traj.append(hs)
            J_traj.append(Js)
        return np.array(h_traj), np.array(J_traj)
    # this should be like a sub function, not sure theres
    # anyway to implement this though, whatever, its a helper anyway!

    def save_show(self, fname):
        if self.save_figs is True:
            path = self.run_dir + fname
            plt.savefig(path, dpi=600)
        else:
            plt.show()
        plt.close()

    def animate_J(self):
        animate2Dtrajectory(self.model_trajectory)
    # decides if to save or plt!
    # I should seperate out the errors :)!

    #(n,) dim numpy arrays
    def relative_error(self, true_model, inferred_model):
        # call it parameters rather than model??
        sqr_diff = np.sum((inferred_model - true_model) ** 2)
        sqr_sum = np.sum(true_model ** 2)
        mean_error = np.mean((inferred_model - true_model) ** 2) ** 0.5
        return mean_error
        # return (sqr_diff / sqr_sum) ** 0.5

    def compute_relative_error(self,):
        # split diagonal and calc errors seperately for both!
        # "gamma" error as defined in Advances in Physics 2017 Nguyen et al.
        # relative construction error.
        h_true, J_true = self.split_diagonal(self.true_model)
        true_parameters = np.append(h_true, J_true)
        h_inferred, J_inferred = self.split_diagonal(self.model_trajectory[-1])
        inferred_parameters = np.append(h_inferred, J_inferred)

        h_err = self.relative_error(h_true, h_inferred)
        J_err = self.relative_error(J_true, J_inferred)
        tot_err = self.relative_error(true_parameters, inferred_parameters)
        print(h_err, J_err, tot_err)
        errors = (true_parameters - inferred_parameters)
        # let's do the histogram!
        # wait this inst quite rigth yet! I want to have array of errors!
        plt.hist(errors, bins=100)
        plt.show()
        # I should print the distribution of these as well cause I think a few
        # contirbute a lot!
        return 0

    def compute_errors(
            self,
            plot_trajectory=False,
            plot_matricies=False,
            plot_correlation=False):
        final_error_matrix = abs(self.true_model - self.model_trajectory[-1])
        error_trajectory = m.error(self.true_model, self.model_trajectory)

        # error_trajectory = m.error(self.true_model, self.model_trajectory)
        # this should maybe all go in its own function!? I might want to have
        # access to this quite often? But I don't want ot have to always calc
        # it right off the bat so maybe not!
        # need to decide what exactly I want this function to do!
        # I need to somehow weight them by importance? Because there is fewer
        # hs? so maybe I need to change my error function from an average to a
        # sum??? things to think on!
        # total error should be only on the upper diagonal anyway!!
        # so theres a few things to think on here!!
        # need to rethink my error measure, I think it should be a
        # sum not a mean! i.e. instead of average its the total error!
        # lets see how that goes!
        # maybe both show interesitng info actually!
        # so on average there is a lower error in the h but it contributes
        # less becasue its so many less!
        h_true, J_true = self.split_diagonal(self.true_model)
        h_traj, J_traj = self.reform_split_trajectory()
        h_errors = m.error(h_true, h_traj)
        J_errors = m.error(J_true, J_traj)

        # error_trajectory = error_trajectory / self.true_model.size
        # h_errors = h_errors / h_true.size
        # J_errors = J_errors / J_true.size
        if plot_matricies is True:
            min_val = np.min(self.true_model)
            max_val = np.max(self.true_model)

            fig, ax = plt.subplots(1, 3)
            ax = ax.ravel()
            ax[0].imshow(
                self.true_model, vmin=min_val, vmax=max_val)
            ax[1].imshow(
                self.model_trajectory[-1], vmin=min_val, vmax=max_val)
            im3 = ax[2].imshow(
                final_error_matrix, vmin=min_val, vmax=max_val)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im3, cax=cbar_ax)
            self.save_show('Aerror_matricies.png')

        if plot_trajectory is True:
            plt.plot(error_trajectory, label='Total Error')
            plt.plot(h_errors, label='h Error')
            plt.plot(J_errors, label='J Error')
            plt.xlabel('n')
            plt.ylabel(r'$\epsilon$')
            plt.legend()
            self.save_show('Aerror_trajectory.png')
        # move this to its own function eventually!
        if plot_correlation is True:
            inferred_model = self.model_trajectory[-1]
            h_inferred, J_inferred = self.split_diagonal(inferred_model)

            inferred_parameters = np.append(h_inferred, J_inferred)
            true_parameters = np.append(h_true, J_true)
            pearson_coef, pearson_pval = pearsonr(
                true_parameters, inferred_parameters)

            plt.plot(
                true_parameters, inferred_parameters, '.',
                label=r'$r_{pcc} = $' + '{:.3f}'.format(pearson_coef))
            # print(np.min(inferred_parameters), np.max(inferred_parameters))
            x = np.linspace(
                np.min(true_parameters), np.max(true_parameters), 50)
            plt.plot(x, x, color='k')
            plt.xlabel(r'$J_{ij} ^{True}$')
            plt.ylabel(r'$J_{ij} ^{Inf}$')
            plt.legend()
            plt.show()
        return error_trajectory

    # defalts to the last step!
    def compute_histogram(
            self, gd_step=-1,
            nbins=100, true_model=False, separate_diagonal=True):

        model = self.model_trajectory[gd_step]
        vmax = np.max(model)
        vmin = np.min(model)
        bins = np.linspace(vmin, vmax, nbins)
        fname = 'Ahistogram_step{}.png'.format(gd_step)

        if true_model is True:
            hs_true, Js_true = self.split_diagonal(self.true_model)
            plt.hist(hs_true, bins=bins, histtype='step')
            plt.hist(Js_true, bins=bins, histtype='step')
            # fname = 'ATruehistogram.png'

        if separate_diagonal is True:
            hs, Js = self.split_diagonal(model)
            print('Means:')
            print(np.mean(hs), np.mean(Js))
            plt.hist(hs, bins=bins, label='hs', alpha=0.5)
            plt.hist(Js, bins=bins, label='Js', alpha=0.5)
            plt.legend()
            self.save_show(fname)

    def compute_raw_values(self, comapre_to_true=True):
        h_traj, J_traj = self.reform_split_trajectory()
        print(h_traj.shape, J_traj.shape)
        for c, hi in enumerate(np.swapaxes(h_traj, 0, 1)):
            plt.plot(hi)
        plt.axhline(self.true_model[0, 0], 0, 1)
        plt.show()

    # def compute_model_correaltion(self):


'''
# specific differences function is what I want to look at as well!
# maybe a distbtuions over time as well!
# maybe I need to have a thing somehwere with all my functions that are
# array operantions and stuff
# definitely need to structure this project more intelgently!
# have a histrogam functioN! with a split argument!
'''
