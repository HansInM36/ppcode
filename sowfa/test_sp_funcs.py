import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
from scipy.interpolate import interp1d
import sliceDataClass as sdc
from funcs import *
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName

sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']
sliceNum = len(sliceList)

var = 'U'
varD = 0 # u:0, v:1, w:2
varName = r'$\rho_{uu}$'
varUnit = ''
varName_save = 'uu_corr'

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']

t_start = 432000.0
t_end = 435600.0
t_delta = 2.0
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


slice = 'Nz0'

readDir = ppDir + '/data/'
readName = slice
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

slc = sdc.Slice(data_org, 2)
H = slc.N_location # height of this plane

tt0 = slc.data['U'][:,0,0]
tt1 = slc.data['U'][:,0,1]


# test nperseg
freq0, Pxy0 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1860, noverlap=None)
freq1, Pxy1 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1314, noverlap=None)
freq2, Pxy2 = scipy.signal.csd(tt0[0:1314], tt0[0:1314], 0.5, nperseg=1314, noverlap=None)

delta_f0 = freq0[1] - freq0[0]
delta_f1 = freq1[1] - freq1[0]
s0 = sum(Pxy0 * delta_f0)
s1 = sum(Pxy1 * delta_f1)
print(s0,s1)

fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(freq0, abs(Pxy0), label='csd, 1860', linewidth=1.0, color='r')
plt.loglog(freq1, abs(Pxy1), label='csd, 1314', linewidth=1.0, color='b')
plt.loglog(freq2, abs(Pxy2), label='csd, 1314', linewidth=1.0, color='g')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
fig.tight_layout() # adjust the layout
plt.show()

# test 0 mean
freq0, Pxy0 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1860, noverlap=None)
freq1, Pxy1 = scipy.signal.csd(tt0-tt0.mean(), tt0-tt0.mean(), 0.5, nperseg=1860, noverlap=None)

fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(freq0, abs(Pxy0), label='csd, 1860', linewidth=1.0, color='r')
plt.loglog(freq1, abs(Pxy1), label='csd, 1024', linewidth=1.0, color='b')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
fig.tight_layout() # adjust the layout
plt.show()

# test noverlap
freq0, Pxy0 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1860, noverlap=10)
freq1, Pxy1 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1860, noverlap=1024)

fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(freq0, abs(Pxy0), label='csd, 1860', linewidth=1.0, color='r')
plt.loglog(freq1, abs(Pxy1), label='csd, 1024', linewidth=1.0, color='b')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
fig.tight_layout() # adjust the layout
plt.show()

# test fft
freq0, Pxy0 = scipy.signal.csd(tt0, tt0, 0.5, nperseg=1860, noverlap=None)
freq1, Pxy1 = PSD_f(tt0-tt0.mean(), 0.5)

delta_f0 = freq0[1] - freq0[0]
delta_f1 = freq1[1] - freq1[0]
s0 = sum(Pxy0 * delta_f0)
s1 = sum(Pxy1 * delta_f1)
print(s0,s1)

fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(freq0, abs(Pxy0), label='csd, 1860', linewidth=1.0, color='r')
plt.loglog(freq1, abs(Pxy1), label='csd, 1024', linewidth=1.0, color='b')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
fig.tight_layout() # adjust the layout
plt.show()

coh = coherence(tt0, tt1, 0.5)
