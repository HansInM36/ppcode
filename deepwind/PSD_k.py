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

def PSD_kx_sowfa(tSeq, ySeq, xSeq, zInd, d, varSeq, segNum):
    tNum = tSeq.size
    xNum = xSeq.size
    yNum = ySeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    psd_list = []
    for t in range(tNum):
        v = varSeq[pInd_start:pInd_end,t]
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(xSeq, v, deg=deg_))
        v_ = v - polyFunc(xSeq)
        # bell tapering
        v_ = funcs.window_weight(v_)
        # FFT
        # omega_seq, tmp = PSD_omega(t_seq,tmp)
        kx_seq, psd = scipy.signal.csd(v_, v_, 2*np.pi/d, nperseg=segNum, noverlap=None)
        psd_list.append(psd)
    psd_seq = np.average(np.array(psd_list), axis=0)
    return kx_seq, psd_seq
def PSD_ky_sowfa(tSeq, ySeq, xSeq, zInd, d, varSeq, segNum):
    tNum = tSeq.size
    xNum = xSeq.size
    yNum = ySeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    psd_list = []
    for t in range(tNum):
        v = varSeq[pInd_start:pInd_end,t]
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(ySeq, v, deg=deg_))
        v_ = v - polyFunc(ySeq)
        # bell tapering
        v_ = funcs.window_weight(v_)
        # FFT
        # omega_seq, tmp = PSD_omega(t_seq,tmp)
        ky_seq, psd = scipy.signal.csd(v_, v_, 2*np.pi/d, nperseg=segNum, noverlap=None)
        psd_list.append(psd)
    psd_seq = np.average(np.array(psd_list), axis=0)
    return ky_seq, psd_seq

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

def PSD_kx_palm(tSeq, xSeq, yInd, zInd, d, varSeq, segNum):
    psd_list = []
    for tInd in range(tSeq.size):
        v = varSeq[tInd,zInd,yInd,:]
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(xSeq, v, deg=deg_))
        v_ = v - polyFunc(xSeq)
        # bell tapering
        v_ = funcs.window_weight(v_)
        kx_seq, psd = scipy.signal.csd(v_, v_, 2*np.pi/d, nperseg=segNum, noverlap=None)
        psd_list.append(psd)
    psd_seq = np.average(np.array(psd_list), axis=0)
    return kx_seq, psd_seq
def PSD_ky_palm(tSeq, ySeq, xInd, zInd, d, varSeq, segNum):
    psd_list = []
    for tInd in range(tSeq.size):
        v = varSeq[tInd,zInd,:,xInd]
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(ySeq, v, deg=deg_))
        v_ = v - polyFunc(ySeq)
        # bell tapering
        v_ = funcs.window_weight(v_)
        ky_seq, psd = scipy.signal.csd(v_, v_, 2*np.pi/d, nperseg=segNum, noverlap=None)
        psd_list.append(psd)
    psd_seq = np.average(np.array(psd_list), axis=0)
    return ky_seq, psd_seq

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0

# tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
# kx_seq_0, psdx_seq_0 = PSD_kx_sowfa(tSeq_0, ySeq_0, xSeq_0, 4, 20, uSeq_0, 40)

tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg3_rf', ((0,0,0),30.0), 'U', 0)
xSeq_0 = xSeq_0[::2]; uSeq_0 = uSeq_0[::2,:] # use parts of the sample points
kx_seq_0, psdx_seq_0 = PSD_kx_sowfa(tSeq_0, ySeq_0, xSeq_0, 1, 10, uSeq_0, 800//10)

tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg4_rf', ((0,0,0),30.0), 'U', 0)
ySeq_0 = ySeq_0[::2]; uSeq_0 = uSeq_0[::2,:] # use parts of the sample points
ky_seq_0, psdy_seq_0 = PSD_ky_sowfa(tSeq_0, ySeq_0, xSeq_0, 1, 10, uSeq_0, 800//10)

# tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg5', ((0,0,0),30.0), 'U', 0)
# xSeq_0 = xSeq_0[::2]; uSeq_0 = uSeq_0[::2,:] # use parts of the sample points
# kx_seq_0, psdx_seq_0 = PSD_kx_sowfa(tSeq_0, ySeq_0, xSeq_0, 1, 10, uSeq_0, 2000//10)
#
# tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg6', ((0,0,0),30.0), 'U', 0)
# ySeq_0 = ySeq_0[::2]; uSeq_0 = uSeq_0[::2,:] # use parts of the sample points
# ky_seq_0, psdy_seq_0 = PSD_ky_sowfa(tSeq_0, ySeq_0, xSeq_0, 1, 10, uSeq_0, 2000//10)
#



# prjDir = '/scratch/palmdata/JOBS'
# jobName  = 'deepwind_NBL'
# dir_1 = prjDir + '/' + jobName
#
# tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M06', ['.004'], 'u')
# kx_seq_1, psdx_seq_1 = PSD_kx_palm(tSeq_1, xSeq_1, 0, 1, 10, uSeq_1, 2000//10)
#
# tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M07', ['.004'], 'u')
# ky_seq_1, psdy_seq_1 = PSD_ky_palm(tSeq_1, ySeq_1, 0, 1, 10, uSeq_1, 2000//10)
#

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_mdf'
dir_1 = prjDir + '/' + jobName

tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M01', ['.002'], 'u')
kx_seq_1, psdx_seq_1 = PSD_kx_palm(tSeq_1, xSeq_1, 0, 1, 10, uSeq_1, 2000//10); del uSeq_1

tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M01', ['.002'], 'u')
ky_seq_1, psdy_seq_1 = PSD_ky_palm(tSeq_1, ySeq_1, 0, 1, 10, uSeq_1, 2000//10); del uSeq_1


""" check PSD """
fig, ax = plt.subplots(figsize=(6,6))
# colors = plt.cm.jet(np.linspace(0,1,6))
colors = ['red','blue','green','salmon','cornflowerblue','lightgreen','k']
dn = 1

plt.loglog(kx_seq_0[0::dn], psdx_seq_0[0::dn], label='sowfa-x', linewidth=1.0, linestyle='-', color='r')
plt.loglog(ky_seq_0[0::dn], psdy_seq_0[0::dn], label='sowfa-y', linewidth=1.0, linestyle='--', color='r')
plt.loglog(kx_seq_1[0::dn], psdx_seq_1[0::dn], label='palm-x', linewidth=1.0, linestyle='-', color='b')
plt.loglog(ky_seq_1[0::dn], psdy_seq_1[0::dn], label='palm-y', linewidth=1.0, linestyle='--', color='b')


# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('k (1/m)', fontsize=12)
plt.ylabel(r'$\mathrm{E_u}$' + ' (' + r'$\mathrm{m^3/s^2}$' + ')', fontsize=12)
xaxis_min = 1e-3
xaxis_max = 0.3 # f_seq.max()
yaxis_min = 1e-3
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(0.04,0.16), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
