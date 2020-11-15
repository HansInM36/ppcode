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

sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7',]
sliceNum = len(sliceList)

var = 'U'
varD = 0 # u:0, v:1, w:2
varName = 'Su'
varUnit = r'$m^2/s$'
varName_save = 'Su'

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

plotDataList = []
HList = []

for slice in sliceList:
    readDir = ppDir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    slc = sdc.Slice(data_org, 2)
    H = slc.N_location # height of this plane
    HList.append(H)

    pNum = slc.data['pNo'].size

    ESD_list = []
    for pInd in range(pNum):
        # interpolate
        vSeq = slc.data[var][:,pInd,varD]
        f = interp1d(tSeq, vSeq)
        v_seq = f(t_seq)
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(t_seq, v_seq, deg=deg_))
        tmp = v_seq - polyFunc(t_seq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = window_weight(tmp)
        # FFT
        omega_seq, tmp = ESD_omega(t_seq,tmp)
        ESD_list.append(tmp)
    ESD_seq = np.average(np.array(ESD_list), axis=0)
    plotDataList.append((omega_seq, ESD_seq))

zNum = len(HList)

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    f_ = plotDataList[zInd][0] / (2*np.pi) # convert from omega to frequency
    ESD_ = plotDataList[zInd][1] * 2*np.pi
    plt.loglog(f_, ESD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
# -5/3 law
plt.loglog(f_[1:], 1e5*np.power(f_[1:], -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('f (1/s)')
plt.ylabel(varName + ' (' + varUnit + ')')
xaxis_min = 1e-4
xaxis_max = omega_seq.max() / (2*np.pi)
yaxis_min = 1e-2
yaxis_max = 1e9
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_ESD_f_' + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
