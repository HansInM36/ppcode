import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
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
varName = r'$Su_x$'
varUnit = r'$m^3/s^2$'
varName_save = 'Su_x'

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']
tInd = -1

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
    slcData = slc.meshITP_Nz((0,2000,100), (0,2000,100), slc.data[var][tInd][:,varD], method_='cubic')
    xSeq = slcData[0]
    xNum = slcData[0].size
    yNum = slcData[1].size

    ESD_list = []
    for yInd in range(yNum):
        # detrend by deg_ order plynomial fit
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(xSeq, slcData[2][yInd], deg=deg_))
        tmp = slcData[2][yInd] - polyFunc(xSeq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = window_weight(tmp)
        kSeq, tmp = ESD_k(xSeq, tmp)
        ESD_list.append(tmp)
    # horizontal average
    ESD_seq = np.average(np.array(ESD_list), axis=0)
    plotDataList.append((kSeq, ESD_seq))

zNum = len(HList)

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    k_ = plotDataList[zInd][0]
    ESD_ = plotDataList[zInd][1]
    plt.loglog(k_, ESD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
# -5/3 law
plt.loglog(k_[1:], 100*np.power(k_[1:], -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('k (1/m)')
plt.ylabel(varName + ' (' + varUnit + ')')
xaxis_min = 1e-3
xaxis_max = kSeq.max()
yaxis_min = 1e-4
yaxis_max = 1e6
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)

plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_ESD_' + str(tSeq[tInd]) + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
