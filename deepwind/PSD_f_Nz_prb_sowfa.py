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

def velo_prb(dir, prbg, trs_para, var, varD):
    """ extract velocity data of specified probe groups """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]

    # read data
    readDir = dir + '/data/'
    readName = prbg
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    coors = data_org['coors']

    # coordinate transformation
    prbNum = coors.shape[0]
    for p in range(prbNum):
        tmp = data_org[var][p]
        data_org[var][p] = funcs.trs(tmp,O,alpha)

    xSeq = np.array(data_org['info'][2])
    ySeq = np.array(data_org['info'][3])
    zSeq = np.array(data_org['info'][4])
    xNum = xSeq.size
    yNum = ySeq.size
    zNum = zSeq.size
    varSeq = data_org[var][:,:,varD]
    tSeq = data_org['time']
    tNum = tSeq.size
    return tSeq, xSeq, ySeq, zSeq, varSeq, coors

def PSD(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    data = varSeq
    tNum = tSeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    # coors = data_org['coors'][xNum*yNum*zInd:xNum*yNum*(zInd+1)]
    # pNum = coors.shape[0]

    PSD_list = []
    for p in range(pInd_start, pInd_end):
        vSeq = varSeq[p]
        # interpolate
        f = interp1d(tSeq, vSeq, kind='linear', fill_value='extrapolate')
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

    return f_seq, PSD_seq


### plot PSD of multi cases at a certain height
zInd = 4

jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = velo_prb(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = velo_prb(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = velo_prb(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq, PSD_u_seq_0 = PSD((144000.0, 146400, 0.1), tSeq_0, zInd, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq, PSD_v_seq_0 = PSD((144000.0, 146400, 0.1), tSeq_0, zInd, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq, PSD_w_seq_0 = PSD((144000.0, 146400, 0.1), tSeq_0, zInd, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)

jobName_1 = 'gs10_0.0001'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1, coors_1 = velo_prb(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1, coors_1 = velo_prb(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1, coors_1 = velo_prb(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq, PSD_u_seq_1 = PSD((43200.0, 45600, 0.1), tSeq_1, zInd, xSeq_1.size, ySeq_1.size, uSeq_1, 20480)
f_seq, PSD_v_seq_1 = PSD((43200.0, 45600, 0.1), tSeq_1, zInd, xSeq_1.size, ySeq_1.size, vSeq_1, 20480)
f_seq, PSD_w_seq_1 = PSD((43200.0, 45600, 0.1), tSeq_1, zInd, xSeq_1.size, ySeq_1.size, wSeq_1, 20480)


# plot
fig, ax = plt.subplots(figsize=(8,5))

plt.loglog(f_seq, PSD_u_seq_0, label='', linewidth=1.0, linestyle='-', color='r')
# plt.loglog(f_seq, PSD_v_seq_0, label='oneEquation-v', linewidth=1.0, linestyle='--', color='r')
# plt.loglog(f_seq, PSD_w_seq_0, label='oneEquation-w', linewidth=1.0, linestyle=':', color='r')
# plt.loglog(f_seq, PSD_u_seq_1, label='standard-u', linewidth=1.0, linestyle='-', color='b')
# plt.loglog(f_seq, PSD_v_seq_1, label='standard-v', linewidth=1.0, linestyle='--', color='b')
# plt.loglog(f_seq, PSD_w_seq_1, label='standard-w', linewidth=1.0, linestyle=':', color='b')

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('Su' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-12
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su' + '_f_prb_' + str(int(zSeq_0[zInd])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()













prbg = 'prbg0'

# coordinate transmation
O = (0,0,0)
alpha = 30.0

var = 'U'
varD = 2 # u:0, v:1, w:2
varName = 'Su'
varUnit = r'$\mathrm{m^2/s}$'
varName_save = 'Su'

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



t_start = 144000.0
t_end = 146400.0
t_delta = 0.1
fs = 1 / t_delta # sampling frequency
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


segNum = 2048

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
fig, ax = plt.subplots(figsize=(8,5))
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
xaxis_max = 1 # f_seq.max()
yaxis_min = 1e-12
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_PSD_f_prb' + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
