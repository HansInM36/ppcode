import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt

# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

prbg = 'prbg0'

# coordinate transmation
O = (0,0,0)
alpha = 30.0

var = 'U'
varD = 2 # u:0, v:1, w:2
varName = 'Su'
varUnit = r'$\mathrm{m^2/s}$'
varName_save = 'Sw'

# read data
readDir = ppDir + '/data/'
readName = prbg
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

# coordinate transformation
prbNum = data_org['coors'].shape[0]
for p in range(prbNum):
    tmp = data_org[var][p]
    data_org[var][p] = funcs.trs(tmp,O,alpha)

# choose the probegroup to be used in plotting
xSeq = np.array(data_org['info'][2])
ySeq = np.array(data_org['info'][3])
zSeq = np.array(data_org['info'][4])
xNum = xSeq.size
yNum = ySeq.size
zNum = zSeq.size

t_start = 432000.0
t_end = 434400.0
t_delta = 0.1
fs = 1 / t_delta # sampling frequency
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


segNum = 1200

plotDataList = []
HList = []

for zInd in range(zNum):

    HList.append(zSeq[zInd])

    data = data_org['U']

    tSeq = data_org['time']
    tNum = tSeq.size

    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    # coors = data_org['coors'][xNum*yNum*zInd:xNum*yNum*(zInd+1)]
    # pNum = coors.shape[0]

    PSD_list = []
    for p in range(pInd_start, pInd_end):
        vSeq = data[p][:,varD]
        # interpolate
        f = interp1d(tSeq, vSeq, fill_value='extrapolate')
        v_seq = f(t_seq)
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(t_seq, v_seq, deg=deg_))
        tmp = v_seq - polyFunc(t_seq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = funcs.window_weight(tmp)
        # FFT
        # omega_seq, tmp = PSD_omega(t_seq,tmp)
        f_seq, tmp = scipy.signal.csd(tmp, tmp, fs, nperseg=segNum, noverlap=None)
        PSD_list.append(tmp)
    PSD_seq = np.average(np.array(PSD_list), axis=0)
    plotDataList.append((f_seq, PSD_seq))


# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    f_ = plotDataList[zInd][0]
    PSD_ = plotDataList[zInd][1]
    plt.loglog(f_, PSD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
# -5/3 law
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel(varName + ' (' + varUnit + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-14
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_PSD_f_prb' + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
