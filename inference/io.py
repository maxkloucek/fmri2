import numpy as np
import h5py


# should say for my MCC stuff this is specicially!
# I need to make sure metadata is always stroed
# i.e. not somehting you edit in the runscript!
class Readhdf5_MC():
    def __init__(self, fname, prod_only=True):
        self.fname = fname
        self.prod_only = prod_only

    def __enter__(self):
        self.fin = h5py.File(self.fname, "r")
        self.show = self.show_contents()

        eq_samples = int(
            self.fin.attrs['EqCycles'] / self.fin.attrs['CycleDumpFreq'])
        if self.prod_only is True:
            self.cuttoff = eq_samples
        if self.prod_only is False:
            self.cuttoff = 0

        return self

    def __exit__(self, *args):
        self.fin.close()

    def show_contents(self):
        print('-- File: "{}" --'.format(self.fname))
        print('Contains the following datasets:')
        for group_name in self.fin.keys():
            dset_names = list(self.fin[group_name].keys())
            print(group_name, dset_names)
        print('\n-- Metadata --')
        for meta_data, value in self.fin.attrs.items():
            print(meta_data, value)
        print('---------------')

    def get_metadata(self):
        metadata_dictionary = {}
        for key, value in self.fin.attrs.items():
            metadata_dictionary[key] = value
        return metadata_dictionary

    def read_single_dataset(self, group_name, dset_name):
        if dset_name == 'energy':
            dataset = self.fin[group_name][dset_name][self.cuttoff:]

        elif dset_name == 'configurations':
            dataset = self.fin[group_name][dset_name][self.cuttoff:, :]
        return dataset

    def read_many_datasets(self, dset_name):
        datasets = []
        for name in self.fin:
            datasets.append(self.read_single_dataset(name, dset_name))
        return np.array(datasets)
