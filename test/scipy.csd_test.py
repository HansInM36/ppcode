import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
import scipy.signal
import funcs
import matplotlib.pyplot as plt


N = int(1800)

x = np.random.rand(N)

# noise
alpha = 1
w = np.random.rand(N) * alpha

y = 0*x + 2 + w

fs = 0.5
nperseg_ = 128

freq, Pxx = scipy.signal.csd(x, x, fs, nperseg=nperseg_, noverlap=None)
freq, Pxy = scipy.signal.csd(x, y, fs, nperseg=nperseg_, noverlap=None)
freq, Pyy = scipy.signal.csd(y, y, fs, nperseg=nperseg_, noverlap=None)
freq = np.squeeze(freq).ravel()

Rxy    = Pxy.real
Qxy    = Pxy.imag
coh    = abs(np.array(Pxy) * np.array(np.conj(Pxy))) / (np.array(Pxx) * np.array(Pyy))
co_coh = np.real(Pxy / np.sqrt(np.array(Pxx) * np.array(Pyy)))
phase  = np.arctan2(Qxy,Rxy)


# freq_, Pxx_ = funcs.PSD_f(x-x.mean(), fs)
freq_, Pxx_ = scipy.signal.csd(x-x.mean(), x-x.mean(), fs, nperseg=nperseg_, noverlap=None)

fig, ax = plt.subplots(figsize=(6,6))
plt.plot(freq, Pxx, linewidth=1.0, color='k')
plt.plot(freq_, Pxx_, linewidth=1.0, color='r')
plt.ylim(0,1.02)
plt.grid()
plt.show()
