import numpy as np
# import matplotlib.pyplot as plt
import scipy.io

from . import rawprocess


def load_fmri(
        data_dir='/Users/mk14423/Desktop/code/0-fmri/data-Kajimura',
        day='01'):
    filename = data_dir + '/ROISignals_Sub_' + day + '.mat'
    mat_contents = scipy.io.loadmat(filename)
    data = mat_contents['ROISignals']  # [:, 0:5]
    z, S, mu, sigma = rawprocess.binarize(data, 0)

    return data, z, S


def get_MMconditions():
    d1s = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    days_MM = [
        '27', '28', '31', '32', '34', '37', '39', '40', '42', '43', '44', '47',
        '48', '50', '51', '55', '56', '58']
    int_days_MM = np.array([int(i) for i in days_MM])
    int_days_noMM = []
    c = 0
    for i in range(0, 59):
        min_val = np.min(int_days_MM)
        if i == min_val:
            int_days_MM[c] = 100
            c += 1
        else:
            int_days_noMM.append(i)
    days_noMM = d1s
    days_noMM.extend([str(j) for j in int_days_noMM[10:]])
    days_all = days_MM + days_noMM
    days_all = sorted(days_all, key=lambda x: int(x))
    return days_noMM, days_MM, days_all


# split data function!
# get full data?!
def load_full_data(days=['01', '02', '03']):
    data = []
    zs = []
    Ss = []
    for day in days:
        raw, z, S = load_fmri(day=day)
        data.append(raw)
        zs.append(z)
        Ss.append(S)
    data = np.array(data)
    zs = np.array(zs)
    Ss = np.array(Ss)
    return data, zs, Ss


def load(condition):
    days_noMM, days_MM, days_all = get_MMconditions()
    if condition == 'noMM':
        days = days_noMM
    elif condition == 'MM':
        days = days_MM
    elif condition == 'all':
        days = days_all
    else:
        print('invalid choice made')
        return 0
    return load_full_data(days=days)
