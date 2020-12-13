import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
import scipy.signal
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']
sliceNum = len(sliceList)

# coordinate transmation
O = (0,0,0)
alpha = 30.0 # rotate the coordinate system to alpha

var = 'U'
varD = 1 # u:0, v:1, w:2
varName = r'$\mathrm{S_v^y}$'
varUnit = r'$\mathrm{m^3/s^2}$'
varName_save = 'Sv_y'

y_delta = 20
k = 2*np.pi/y_delta

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']
tInd = -1

segNum = 64

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

    tmp = slc.data[var][tInd][:]
    tmp = funcs.trs(tmp,O,alpha)

    slcData = slc.meshITP_Nz((0,2000,100), (0,2000,100), tmp[:,varD], method_='linear')
    ySeq = slcData[1]
    yNum = slcData[1].size
    xNum = slcData[0].size

    PSD_list = []
    for xInd in range(xNum):
        # detrend by deg_ order plynomial fit
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(ySeq, slcData[2][xInd], deg=deg_))
        tmp = slcData[2][xInd] - polyFunc(ySeq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = funcs.window_weight(tmp)
        # FFT
        # k_seq, tmp = funcs.PSD_k(tmp, 1/x_delta)
        k_seq, tmp = scipy.signal.csd(tmp, tmp, k/2/np.pi, nperseg=segNum, noverlap=None)
        k_seq *= 2*np.pi
        tmp *= 1/2/np.pi
        PSD_list.append(tmp)
    # horizontal average
    PSD_seq = np.average(np.array(PSD_list), axis=0)
    plotDataList.append((k_seq, PSD_seq))

zNum = len(HList)

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    k_ = plotDataList[zInd][0]
    PSD_ = plotDataList[zInd][1]
    plt.loglog(k_, PSD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
# # -5/3 law
# plt.loglog(k_[1:], 100*np.power(k_[1:], -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('k (1/m)')
plt.ylabel(varName + ' (' + varUnit + ')')
xaxis_min = 1e-3
xaxis_max = k_seq.max()
# yaxis_min = 1e-8
# yaxis_max = 1e2
# plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_PSD_' + str(tSeq[tInd]) + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
