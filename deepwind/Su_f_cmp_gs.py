import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt


def getData_sowfa(dir, prbg, trs_para, var, varD):
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

def PSD_sowfa(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

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

def getData_palm(dir, jobName, maskID, run_no_list, var):
    """ extract velocity data of specified probe groups """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq

def PSD_palm(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    # fs = 10
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    PSD_list = []
    for yInd in range(yNum):
        for xInd in range(xNum):
            vSeq = varSeq[:,zInd,yInd,xInd]
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


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_0, PSD_u_seq_0 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
# f_seq_0, PSD_u_seq_0 = PSD_sowfa((144000.0, 145200, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, 10240)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs10'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.022'], 'u')
f_seq_1, PSD_u_seq_1 = PSD_palm((288000.0, 290400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, varSeq_1, 20480)
# f_seq_1, PSD_u_seq_1 = PSD_palm((288000.0, 289200, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, varSeq_1, 10240)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs20'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_2, PSD_u_seq_2 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 2, xSeq_2.size, ySeq_2.size, uSeq_2, 20480)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs20'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, varSeq_3 = getData_palm(dir_3, jobName, 'M03', ['.002','.003','.004'], 'u')
f_seq_3, PSD_u_seq_3 = PSD_palm((432000.0, 434400, 0.1), tSeq_3, 3, xSeq_3.size, ySeq_3.size, varSeq_3, 20480)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_4 = 'gs40'
ppDir_4 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_4
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4, coors_4 = getData_sowfa(ppDir_4, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_4, PSD_u_seq_4 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_4, 2, xSeq_4.size, ySeq_4.size, uSeq_4, 20480)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs40'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, varSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.001'], 'u')
f_seq_5, PSD_u_seq_5 = PSD_palm((144000.0, 146400, 0.1), tSeq_5, 1, xSeq_5.size, ySeq_5.size, varSeq_5, 20480)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5'
dir_6 = prjDir + '/' + jobName
tSeq_6, xSeq_6, ySeq_6, zSeq_6, varSeq_6 = getData_palm(dir_6, jobName, 'M03', ['.011','.012','.013','.014'], 'u')
f_seq_6, PSD_u_seq_6 = PSD_palm((72000.0, 74400, 0.1), tSeq_6, 4, xSeq_6.size, ySeq_6.size, varSeq_6, 20480)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs40_nogt'
dir_7 = prjDir + '/' + jobName
tSeq_7, xSeq_7, ySeq_7, zSeq_7, varSeq_7 = getData_palm(dir_7, jobName, 'M03', ['.000','.001'], 'u')
f_seq_7, PSD_u_seq_7 = PSD_palm((144000.0, 146400, 0.1), tSeq_7, 4, xSeq_7.size, ySeq_7.size, varSeq_7, 20480)


# plot
fig, ax = plt.subplots(figsize=(5.2,3))
colors = plt.cm.jet(np.linspace(0,1,8))

# plt.loglog(f_seq_0, PSD_u_seq_0, label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors[0])
# plt.loglog(f_seq_1, PSD_u_seq_1, label='palm-gs10', linewidth=1.0, linestyle='-', color=colors[1])
# plt.loglog(f_seq_2, PSD_u_seq_2, label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors[2])
# plt.loglog(f_seq_3, PSD_u_seq_3, label='palm-gs20', linewidth=1.0, linestyle='-', color=colors[3])
# plt.loglog(f_seq_4, PSD_u_seq_4, label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors[4])
plt.loglog(f_seq_5, PSD_u_seq_5, label='palm-gs40', linewidth=1.0, linestyle='-', color=colors[5])
# plt.loglog(f_seq_6, PSD_u_seq_6, label='palm-gs5', linewidth=1.0, linestyle='-', color=colors[6])
plt.loglog(f_seq_7, PSD_u_seq_7, label='palm-gs40_nogt', linewidth=1.0, linestyle='-', color=colors[7])
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('Su' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e3
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.02,0.42), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
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
