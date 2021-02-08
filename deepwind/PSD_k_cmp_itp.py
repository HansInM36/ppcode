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

def PSD_kx_palm(tSeq, xSeq, zInd, d, varSeq, segNum):
    psd_list = []
    for tInd in range(tSeq.size):
        v = varSeq[tInd,zInd,0,:]
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
def PSD_ky_palm(tSeq, ySeq, zInd, d, varSeq, segNum):
    psd_list = []
    for tInd in range(tSeq.size):
        v = varSeq[tInd,zInd,:,0]
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
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0

# tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir, 'prbg0', ((0,0,0),30.0), 'U', 0)
# kx_seq_0, psdx_seq_0 = PSD_kx_sowfa(tSeq_0, ySeq_0, xSeq_0, 4, 20, uSeq_0, 40)

tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3, coors_3 = getData_sowfa(ppDir, 'prbg3', ((0,0,0),30.0), 'U', 0)
# xSeq_3 = xSeq_3[::2]; uSeq_3 = uSeq_3[::2,:] # use parts of the sample points
kx_seq_3, psdx_seq_3 = PSD_kx_sowfa(tSeq_3, ySeq_3, xSeq_3, 1, 5, uSeq_3, 2000//5)

tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4, coors_4 = getData_sowfa(ppDir, 'prbg4', ((0,0,0),30.0), 'U', 0)
ky_seq_4, psdy_seq_4 = PSD_ky_sowfa(tSeq_4, ySeq_4, xSeq_4, 1, 5, uSeq_4, 2000//5)

tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5, coors_5 = getData_sowfa(ppDir, 'prbg5', ((0,0,0),0.0), 'U', 0)
# xSeq_5 = xSeq_5[::2]; uSeq_5 = uSeq_5[::2,:] # use parts of the sample points
kx_seq_5, psdx_seq_5 = PSD_kx_sowfa(tSeq_5, ySeq_5, xSeq_5, 1, 5, uSeq_5, 2000//5)

tSeq_6, xSeq_6, ySeq_6, zSeq_6, uSeq_6, coors_6 = getData_sowfa(ppDir, 'prbg6', ((0,0,0),0.0), 'U', 0)
ky_seq_6, psdy_seq_6 = PSD_ky_sowfa(tSeq_6, ySeq_6, xSeq_6, 1, 5, uSeq_6, 2000//5)

tSeq_7, xSeq_7, ySeq_7, zSeq_7, uSeq_7, coors_7 = getData_sowfa(ppDir, 'prbg7', ((0,0,0),30.0), 'U', 0)
kx_seq_7, psdx_seq_7 = PSD_kx_sowfa(tSeq_7, ySeq_7, xSeq_7, 1, 5, uSeq_7, 2000//5)

tSeq_8, xSeq_8, ySeq_8, zSeq_8, uSeq_8, coors_8 = getData_sowfa(ppDir, 'prbg8', ((0,0,0),30.0), 'U', 0)
kx_seq_8, psdx_seq_8 = PSD_kx_sowfa(tSeq_8, ySeq_8, xSeq_8, 1, 5, uSeq_8, 2000//5)

tSeq_9, xSeq_9, ySeq_9, zSeq_9, uSeq_9, coors_9 = getData_sowfa(ppDir, 'prbg9', ((0,0,0),30.0), 'U', 0)
kx_seq_9, psdx_seq_9 = PSD_kx_sowfa(tSeq_9, ySeq_9, xSeq_9, 1, 5, uSeq_9, 2000//5)

tSeq_10, xSeq_10, ySeq_10, zSeq_10, uSeq_10, coors_10 = getData_sowfa(ppDir, 'prbg10', ((0,0,0),30.0), 'U', 0)
ky_seq_10, psdy_seq_10 = PSD_ky_sowfa(tSeq_10, ySeq_10, xSeq_10, 1, 5, uSeq_10, 2000//5)

tSeq_11, xSeq_11, ySeq_11, zSeq_11, uSeq_11, coors_11 = getData_sowfa(ppDir, 'prbg11', ((0,0,0),30.0), 'U', 0)
ky_seq_11, psdy_seq_11 = PSD_ky_sowfa(tSeq_11, ySeq_11, xSeq_11, 1, 5, uSeq_11, 2000//5)

tSeq_12, xSeq_12, ySeq_12, zSeq_12, uSeq_12, coors_12 = getData_sowfa(ppDir, 'prbg12', ((0,0,0),30.0), 'U', 0)
ky_seq_12, psdy_seq_12 = PSD_ky_sowfa(tSeq_12, ySeq_12, xSeq_12, 1, 5, uSeq_12, 2000//5)


""" check PSD_kx """
fig, ax = plt.subplots(figsize=(6,4))
dn = 1
plt.loglog(kx_seq_3[0::dn], psdx_seq_3[0::dn], label='cell', linewidth=1.0, linestyle='-', color='r')
plt.loglog(kx_seq_7[0::dn], psdx_seq_7[0::dn], label='cellPoint', linewidth=1.0, linestyle='-', color='b')
plt.loglog(kx_seq_8[0::dn], psdx_seq_8[0::dn], label='cellPointFace', linewidth=1.0, linestyle='-', color='g')
plt.loglog(kx_seq_9[0::dn], psdx_seq_9[0::dn], label='pointMVC', linewidth=1.0, linestyle='-', color='y')

plt.loglog(kx_seq_5[0::dn], psdx_seq_5[0::dn], label='cellPointFace-0deg', linewidth=1.0, linestyle='-', color='k')

plt.vlines(0.314, 1e-8, 1e2, linestyle='--', color='k')
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel(r'$\mathrm{k_x}$ (1/m)')
plt.ylabel('S' + ' (' + r'$\mathrm{m^3/s^2}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1.0 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'kx_cmp_itp.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD' + '/' + saveName, bbox_inches='tight')
plt.show()

""" check PSD_ky """
fig, ax = plt.subplots(figsize=(6,4))
dn = 1
plt.loglog(ky_seq_4[0::dn], psdy_seq_4[0::dn], label='cell', linewidth=1.0, linestyle='-', color='r')
plt.loglog(ky_seq_10[0::dn], psdy_seq_10[0::dn], label='cellPoint', linewidth=1.0, linestyle='-', color='b')
plt.loglog(ky_seq_11[0::dn], psdy_seq_11[0::dn], label='cellPointFace', linewidth=1.0, linestyle='-', color='g')
plt.loglog(ky_seq_12[0::dn], psdy_seq_12[0::dn], label='pointMVC', linewidth=1.0, linestyle='-', color='y')

plt.loglog(ky_seq_6[0::dn], psdy_seq_6[0::dn], label='cellPointFace-0deg', linewidth=1.0, linestyle='-', color='k')

plt.vlines(0.314, 1e-8, 1e2, linestyle='--', color='k')
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel(r'$\mathrm{k_x}$ (1/m)')
plt.ylabel('S' + ' (' + r'$\mathrm{m^3/s^2}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1.0 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'ky_cmp_itp.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD' + '/' + saveName, bbox_inches='tight')
plt.show()
