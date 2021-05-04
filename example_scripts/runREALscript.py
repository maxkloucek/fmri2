import numpy as np
import matplotlib.pyplot as plt

import inference.sweep as s
import inference.core.preprocess as pp
# rom inference.core.binarise import load_fmri
from inference.core import measures as m
# read in data and do stuff with it?!

days_noMM, days_MM, days = pp.get_MMconditions()
'''
data_spin_matrix = pp.average_spin_matrix(days_noMM)
plt.imshow(data_spin_matrix)
plt.colorbar()
plt.show()

plt.hist(data_spin_matrix.ravel(), bins=100)
plt.show()
print(days_noMM[0])
'''
test_data, test_z, test_S, = pp.load_fmri(day=days_noMM[0])
# plot as a function of tw?
autocorr = m.self_correlation(test_S, 1)
plt.semilogx(np.arange(1, autocorr.size+1), autocorr, 'x-')
plt.xlabel('t')
plt.ylabel('Autocorr')
plt.show()
# this is fine for now I think!!! Glad to have it like this
# should look at the distbrution somehow!
# think this is a nice place to laeve it!
# I can work on a subset of this matrix (e.g. the first 10 spins) to 
# get a feeling for my measures!
# make it run on blue crystla?! -> Yushi
# make test it on a subset for now!
# I should intialise my system size, based on the size of
# data_spin_matrix!!!!
'''
obsMM = []
for day in days_MM:
    print(day)
    data, z, s_trajectory = pp.load_fmri(day=day)
    si_data, sij_data, si_sj_data = m.correlations(s_trajectory)
    
    obsMM.append(data_spin_matrix[1, 1])
    # lets just look at some random histrograms to see whatsup!
    # plt.imshow(data_spin_matrix)
    # plt.show()

obsnoMM = []
for day in days_noMM:
    print(day)
    data, z, s_trajectory = pp.load_fmri(day=day)
    si_data, sij_data, si_sj_data = m.correlations(s_trajectory)
    data_spin_matrix = s.gen_spin_matrix(si_data, sij_data)
    obsnoMM.append(data_spin_matrix[1, 1])
    # lets just look at some random histrograms to see whatsup!
    # plt.imshow(data_spin_matrix)
    # plt.show()

obs = []
for day in days:
    print(day)
    data, z, s_trajectory = pp.load_fmri(day=day)
    si_data, sij_data, si_sj_data = m.correlations(s_trajectory)
    data_spin_matrix = s.gen_spin_matrix(si_data, sij_data)
    obs.append(data_spin_matrix[1, 1])
    # lets just look at some random histrograms to see whatsup!
    # plt.imshow(data_spin_matrix)
    # plt.show()

plt.hist(obs)
plt.hist(obsnoMM)
plt.hist(obsMM)
plt.show()
# just going in blind would be a bit rouge, but how can I know?!
'''