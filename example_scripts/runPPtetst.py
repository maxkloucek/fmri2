import numpy as np
import matplotlib.pyplot as plt

import inference.preprocess as pp
import inference.tools as tools
plt.style.use('~/Devel/styles/custom.mplstyle')

all_raw, _, _ = pp.load('all')
reshaped_data = pp.reshape_multiday_data(all_raw)

example_data = all_raw[0, :, 66]
print(example_data.shape)

feq_ROI1, spectrum_ROI1, spec_sdv_ROI1 = tools.fourier_transform_multi_series(
    [example_data])

freq_day1, spectrum_day1, spec_sdv_day1 = tools.fourier_transform_multi_series(
    all_raw[0].T)

freq_all, spec_all, spec_sdv_all = tools.fourier_transform_multi_series(
    reshaped_data
)
peak_index = np.argmax(spec_all)
peak_freq = freq_all[peak_index]
characteristic_period = 1 / peak_freq
print(peak_freq, characteristic_period)

fig, ax = plt.subplots(2, 1)
ax = ax.ravel()

ax[0].plot(
    example_data,
    label=r'$f_{peak}=$' + '{:.2f}'.format(peak_freq) + '\n' +
    r'$\tau=$' + '{:.2f}'.format(characteristic_period))
nlines = int(example_data.size / characteristic_period)
for n in range(0, nlines):
    ax[0].axvline(characteristic_period * n, marker=',')

ax[0].set(xlabel=r'$t$ (index)', ylabel=r'$S(t)$')
ax[0].legend()

ax[1].plot(feq_ROI1, spectrum_ROI1, label='Example; Day1, ROI1')
ax[1].errorbar(freq_all, spec_all, yerr=spec_sdv_all, label='Full data Mean')
# ax.plot(freq_day1, spectrum_day1, label='DAY 1 Mean')
y_lower = spec_all - spec_sdv_all
y_upper = spec_all + spec_sdv_all
ax[1].fill_between(freq_day1, y_lower, y_upper, color='lightgray', alpha=0.8)
ax[1].axvline(peak_freq)
ax[1].set(
    xlabel=r'$f$ (index$^{-1}$)',
    ylabel=r'$\widehat{S} (f)$', yscale='linear')
ax[1].legend()
plt.tight_layout()
plt.show()

'''
from scipy.fft import ifft
yinv = ifft(spectrum_ROI1)
plt.plot(yinv, marker='.')
plt.plot(raw_data[:, 0], marker='.')
plt.show()
'''
# print(all_raw.shape)
# raw_spin_traj = raw_data[:, 0]
# raw_spin_traj = all_raw.ravel()
# pp.yushis_eq_test(raw_spin_traj)

# plt.imshow(configs)
# plt.show()
# all_data = raw_data[:, 0].ravel()

# plt.plot(z[:, 0])
# plt.plot(configs[:, 0])
# plt.show()
# print(configs[:, 0])
# pp.histogram(all_data)
# plt.show()


# noMM_raw, noMM_z, _ = pp.load('noMM')
# MM_raw, MM_z, _ = pp.load('MM')
# pp.histogram(MM_raw[0, :, :], 200)
# pp.histogram(MM_raw, 200)
# all_raw, all_z, _ = pp.load('all')
# pp.histogram(all_raw, 200)
# pp.histogram(all_z, 200)
# print(MM_raw.shape)
# I've done this wrong, I wnat to loop over ROIS!
'''
# trajs = np.array([MM_raw[0, :, i] for i in range(0, 399)])
ths_all = []
for raw_day in MM_raw:
    # ths = pp.kmeans_thresholds(MM_raw[0, :, :])
    ths = pp.kmeans_thresholds(raw_day)
    ths_all.append(ths)
ths_all = np.array(ths_all).ravel()
plt.hist(ths_all, 200)
plt.show()
'''
# min and max are the thingies!
'''
ROI_choice = 99
data = MM_raw[:, :, ROI_choice].ravel()
nbins = 400
pp.histogram(data, nbins)
'''
# lets do it for 1 day & each spin!
# bins = np.linspace(-8, 8, nbins)
# should I do a plot of the cutoffs by clustering;
# sure this sounds interesting!
