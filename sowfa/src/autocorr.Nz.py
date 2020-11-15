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

    autocorr_list = []
    for pInd in range(pNum):
        # interpolate
        vSeq = slc.data[var][:,pInd,varD]
        f = interp1d(tSeq, vSeq, fill_value='extrapolate')
        v_seq = f(t_seq)

        tau_seq, tmp = corr(t_seq, v_seq, v_seq)
        autocorr_list.append(tmp)
    autocorr_seq = np.average(np.array(autocorr_list), axis=0)
    plotDataList.append((tau_seq, autocorr_seq))

zNum = len(HList)

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    tau_ = plotDataList[zInd][0] # convert from omega to frequency
    autocorr_ = plotDataList[zInd][1]
    plt.plot(tau_, autocorr_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(varName)
xaxis_min = 0.0
xaxis_max = 1800.0
xaxis_d = 200.0
yaxis_min = -0.8
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
