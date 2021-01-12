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

prjDir = '/scratch/palmdata/JOBS'
jobName_0  = 'wwinta_0'
dir_0 = prjDir + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0 = getData_palm(dir_0, jobName_0, 'M03', ['.001'], 'u')
f_seq_0, PSD_u_seq_0 = PSD_palm((144000.0, 146400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, varSeq_0, 20480)

prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'wwinta_1'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName_1, 'M03', ['.001'], 'u')
f_seq_1, PSD_u_seq_1 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, varSeq_1, 20480)

prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'mini'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName_1, 'M03', ['.001'], 'u')
f_seq_1, PSD_u_seq_1 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, varSeq_1, 20480)

prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'wwinta_2'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName_1, 'M03', ['.000'], 'u')
f_seq_1, PSD_u_seq_1 = PSD_palm((72000.0, 74400, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, varSeq_1, 4000)

prjDir = '/scratch/palmdata/JOBS'
jobName_2  = 'wwinta_2'
dir_2 = prjDir + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, varSeq_2 = getData_palm(dir_2, jobName_2, 'M03', ['.000'], 'u')
f_seq_2, PSD_u_seq_2 = PSD_palm((72000.0, 74400, 0.5), tSeq_2, 1, xSeq_2.size, ySeq_2.size, varSeq_2, 4000)

prjDir = '/scratch/palmdata/JOBS'
jobName_3  = 'wwinta_2'
dir_3 = prjDir + '/' + jobName_3
tSeq_3, xSeq_3, ySeq_3, zSeq_3, varSeq_3 = getData_palm(dir_3, jobName_3, 'M03', ['.000'], 'u')
f_seq_3, PSD_u_seq_3 = PSD_palm((72000.0, 74400, 0.5), tSeq_3, 2, xSeq_3.size, ySeq_3.size, varSeq_3, 4000)

prjDir = '/scratch/palmdata/JOBS'
jobName_4  = 'wwinta_2'
dir_4 = prjDir + '/' + jobName_4
tSeq_4, xSeq_4, ySeq_4, zSeq_4, varSeq_4 = getData_palm(dir_4, jobName_4, 'M03', ['.000'], 'u')
f_seq_4, PSD_u_seq_4 = PSD_palm((72000.0, 74400, 0.5), tSeq_4, 3, xSeq_4.size, ySeq_4.size, varSeq_4, 4000)

prjDir = '/scratch/palmdata/JOBS'
jobName_5  = 'wwinta_2'
dir_5 = prjDir + '/' + jobName_5
tSeq_5, xSeq_5, ySeq_5, zSeq_5, varSeq_5 = getData_palm(dir_5, jobName_5, 'M03', ['.000'], 'u')
f_seq_5, PSD_u_seq_5 = PSD_palm((72000.0, 74400, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, varSeq_5, 4000)

# plot
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.jet(np.linspace(0,1,8))

plt.loglog(f_seq_0, PSD_u_seq_0, label='no wave', linewidth=1.0, linestyle='-', color='k')
plt.loglog(f_seq_1, PSD_u_seq_1, label='coupling-10m', linewidth=1.0, linestyle='--', color=colors[0])
plt.loglog(f_seq_2, PSD_u_seq_2, label='coupling-30m', linewidth=1.0, linestyle='--', color=colors[1])
plt.loglog(f_seq_3, PSD_u_seq_3, label='coupling-50m', linewidth=1.0, linestyle='--', color=colors[2])
plt.loglog(f_seq_4, PSD_u_seq_4, label='coupling-70m', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_5, PSD_u_seq_5, label='coupling-90m', linewidth=1.0, linestyle='--', color=colors[4])
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
