# import numpy as np
import h5py
import inference.preprocess as pp
import matplotlib.pyplot as plt
plt.style.use('~/Devel/styles/custom.mplstyle')

# writing
# data, z, s = pp.load_fmri()

data, z, s = pp.load('all')
# print(data.shape)
# this is how to save meta data!! I like it!

with h5py.File("full_dataset_TH0.hdf5", "w") as f:
    raw_dset = f.create_dataset("raw-signal", data=data)
    raw_dset[()] = data
    z_dset = f.create_dataset("z-signal", data=z)
    z_dset[()] = z
    s_dset = f.create_dataset("configurations", data=s)
    s_dset[()] = s
    f.attrs['meta'] = 69.420

# reading
# I can cut and read in as appropirate like this!!
infile = h5py.File('full_dataset_TH0.hdf5', 'r')
print(infile)
print(infile.keys())
print('-----')
for meta_data, value in infile.attrs.items():
    print(meta_data, value)

print(infile.items())
# print(dataset[:])
print(infile["raw-signal"][()].shape)
read_data = infile["raw-signal"][0, :, 0]
print(read_data.shape)
plt.plot(read_data)
# plt.plot(read_data[0, :, 0])
plt.show()
# I want to summ over all the days
# remember to close or use a with block?
