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

def crosscorr_sowfa(t_para, tSeq, dInd, zInd, xNum, yNum, vSeq):
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)-dInd

    crosscorr_list = []
    for p in range(pInd_start,pInd_end):
        v0 = vSeq[p]
        v1 = vSeq[p+dInd]
        f0 = interp1d(tSeq, v0, kind='linear', fill_value='extrapolate')
        v0_ = f0(t_seq)
        f1 = interp1d(tSeq, v1, kind='linear', fill_value='extrapolate')
        v1_ = f1(t_seq)
        tau, corr, phase = funcs.crosscorr_FFT(v1_, v0_, 1/fs, norm_=True)
        crosscorr_list.append(corr)

    corr = np.average(np.array(crosscorr_list), axis=0)
    return tau, corr

def crosscorr_palm(t_para, tSeq, dInd, zInd, xNum, vSeq):
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    crosscorr_list = []
    for p in range(0,xNum-dInd):
        v0 = vSeq[:,zInd,0,p]
        v1 = vSeq[:,zInd,0,p+dInd]
        f0 = interp1d(tSeq, v0, kind='linear', fill_value='extrapolate')
        v0_ = f0(t_seq)
        f1 = interp1d(tSeq, v1, kind='linear', fill_value='extrapolate')
        v1_ = f1(t_seq)
        tau, corr, phase = funcs.crosscorr_FFT(v1_, v0_, 1/fs, norm_=True)
        crosscorr_list.append(corr)

    corr = np.average(np.array(crosscorr_list), axis=0)
    return tau, corr

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tau_0, corr_0 = crosscorr_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, 4, xSeq_0.size, ySeq_0.size, uSeq_0)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.001','.002'], 'u')
tau_1, corr_1 = crosscorr_palm((144000.0, 146400, 0.1), tSeq_1, 4, 4, xSeq_1.size, uSeq_1)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3 = getData_palm(dir_3, jobName, 'M01', ['.002','.003','.004'], 'u')
tau_3, corr_3 = crosscorr_palm((144000.0, 146400, 0.1), tSeq_3, 4, 0, xSeq_3.size, uSeq_3)

fig, ax = plt.subplots(figsize=(8,6))
plt.plot(tau_0, corr_0, label='sowfa-80m', linewidth=1.0, color='r')
plt.plot(tau_1, corr_1, label='palm-80m', linewidth=1.0, color='b')
plt.plot(tau_3, corr_3, label='palm-80m', linewidth=1.0, color='g')
plt.xlabel(r'$\mathrm{\tau}$ (s)')
plt.ylabel(r"$\mathrm{\rho_{cross}}$")
xaxis_min = 0.0
xaxis_max = 120.0
xaxis_d = 10.0
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.6,0.8), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName_save + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()























prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind'
suffix = '_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M03'

cycle_no_list = ['.002','.003','.004'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

var = 'u'
varName = r'$\mathrm{\rho_{uu}}$'
varUnit = ''
varName_save = 'uu_corr'


# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
varSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

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
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

t_start = 432000.0
t_end = 434400.0
t_delta = 0.1
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


plotDataList = []
HList = []

for zInd in range(1,zNum): # if zSeq[0] = 0, the data should not be used
    HList.append(zSeq[zInd])

    pNum = int(xNum*yNum)

    autocorr_list = []
    for yInd in range(yNum):
        for xInd in range(xNum):
            vSeq = varSeq[:,zInd,yInd,xInd]
            # interpolate
            f = interp1d(tSeq, vSeq, fill_value='extrapolate')
            v_seq = f(t_seq)
            # detrend
            v_seq = funcs.detrend(t_seq, v_seq)
            tau_seq, tmp = funcs.corr(t_seq, v_seq, v_seq)
            autocorr_list.append(tmp)
    autocorr_seq = np.average(np.array(autocorr_list), axis=0)
    plotDataList.append((tau_seq, autocorr_seq))


# plot

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,9))

for i in range(9):
    zInd = i+1
    tau_ = plotDataList[zInd][0] # convert from omega to frequency
    autocorr_ = plotDataList[zInd][1]
    plt.plot(tau_, autocorr_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[i])
plt.xlabel(r'$\mathrm{\tau}$ (s)')
plt.ylabel(varName)
xaxis_min = 0.0
xaxis_max = 1200.0
xaxis_d = 200.0
yaxis_min = -0.8
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
