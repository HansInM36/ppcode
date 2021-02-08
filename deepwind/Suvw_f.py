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
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)
f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)
f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs20'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, 20480)
f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, 20480)
f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)
f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)
f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, 20480)



prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.001','.002'], 'u')
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.001','.002'], 'v')
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.001','.002'], 'w')
f_seq_1, PSD_u_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, uSeq_1, 20480)
f_seq_1, PSD_u_seq_1_100 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, 20480)
f_seq_1, PSD_u_seq_1_180 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, uSeq_1, 20480)
f_seq_1, PSD_v_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, vSeq_1, 20480)
f_seq_1, PSD_v_seq_1_100 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, vSeq_1, 20480)
f_seq_1, PSD_v_seq_1_180 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, vSeq_1, 20480)
f_seq_1, PSD_w_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, wSeq_1, 20480)
f_seq_1, PSD_w_seq_1_100 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, wSeq_1, 20480)
f_seq_1, PSD_w_seq_1_180 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, wSeq_1, 20480)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5'
dir_2 = prjDir + '/' + jobName
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2 = getData_palm(dir_2, jobName, 'M03', ['.011','.012','.013','.014'], 'u')
tSeq_2, xSeq_2, ySeq_2, zSeq_2, vSeq_2 = getData_palm(dir_2, jobName, 'M03', ['.011','.012','.013','.014'], 'v')
tSeq_2, xSeq_2, ySeq_2, zSeq_2, wSeq_2 = getData_palm(dir_2, jobName, 'M03', ['.011','.012','.013','.014'], 'w')
f_seq_2, PSD_u_seq_2_20 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, uSeq_2, 20480)
f_seq_2, PSD_u_seq_2_100 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, uSeq_2, 20480)
f_seq_2, PSD_u_seq_2_180 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, uSeq_2, 20480)
f_seq_2, PSD_v_seq_2_20 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, vSeq_2, 20480)
f_seq_2, PSD_v_seq_2_100 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, vSeq_2, 20480)
f_seq_2, PSD_v_seq_2_180 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, vSeq_2, 20480)
f_seq_2, PSD_w_seq_2_20 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, wSeq_2, 20480)
f_seq_2, PSD_w_seq_2_100 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, wSeq_2, 20480)
f_seq_2, PSD_w_seq_2_180 = PSD_palm((72000.0, 74400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, wSeq_2, 20480)





""" test cases """

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u8_nbl'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M01', ['.001','.002','.003'], 'u')
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1 = getData_palm(dir_1, jobName, 'M01', ['.001','.002','.003'], 'v')
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1 = getData_palm(dir_1, jobName, 'M01', ['.001','.002','.003'], 'w')
f_seq_1, PSD_u_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, uSeq_1, 20480)
f_seq_1, PSD_v_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, vSeq_1, 20480)
f_seq_1, PSD_w_seq_1_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, wSeq_1, 20480)
ua_1 = uSeq_1.mean()
var_1 = uSeq_1.var()


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_cbl'
dir_2 = prjDir + '/' + jobName
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2 = getData_palm(dir_2, jobName, 'M01', ['.002','.004','.005'], 'u')
tSeq_2, xSeq_2, ySeq_2, zSeq_2, vSeq_2 = getData_palm(dir_2, jobName, 'M01', ['.002','.004','.005'], 'v')
tSeq_2, xSeq_2, ySeq_2, zSeq_2, wSeq_2 = getData_palm(dir_2, jobName, 'M01', ['.002','.004','.005'], 'w')
f_seq_2, PSD_u_seq_2_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, uSeq_2, 20480)
f_seq_2, PSD_v_seq_2_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, vSeq_2, 20480)
f_seq_2, PSD_w_seq_2_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, wSeq_2, 20480)
ua_2 = uSeq_2[:,4,:,:].mean()
var_2 = uSeq_2.var()

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3 = getData_palm(dir_3, jobName, 'M01', ['.002','.003','.004'], 'u')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, vSeq_3 = getData_palm(dir_3, jobName, 'M01', ['.002','.003','.004'], 'v')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, wSeq_3 = getData_palm(dir_3, jobName, 'M01', ['.002','.003','.004'], 'w')
f_seq_3, PSD_u_seq_3_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, uSeq_3, 20480)
f_seq_3, PSD_v_seq_3_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, vSeq_3, 20480)
f_seq_3, PSD_w_seq_3_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, wSeq_3, 20480)
ua_3 = uSeq_3.mean()
var_3 = uSeq_3.var()

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl_1'  # based on 'pcr_u12_cbl', set heatflux to 0, back ground pressure to 0, turn on coriolis force
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4 = getData_palm(dir_4, jobName, 'M01', ['.001','.002','.003'], 'u')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, vSeq_4 = getData_palm(dir_4, jobName, 'M01', ['.001','.002','.003'], 'v')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, wSeq_4 = getData_palm(dir_4, jobName, 'M01', ['.001','.002','.003'], 'w')
f_seq_4, PSD_u_seq_4_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_4, 0, xSeq_4.size, ySeq_4.size, uSeq_4, 20480)
f_seq_4, PSD_v_seq_4_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_4, 0, xSeq_4.size, ySeq_4.size, vSeq_4, 20480)
f_seq_4, PSD_w_seq_4_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_4, 0, xSeq_4.size, ySeq_4.size, wSeq_4, 20480)
ua_4 = uSeq_4.mean()
var_4 = uSeq_4.var()

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl_2' # based on 'pcr_u12_nbl_1', change ug, vg so that u90 = 8
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = getData_palm(dir_5, jobName, 'M01', ['.001','.002','.003'], 'u')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, vSeq_5 = getData_palm(dir_5, jobName, 'M01', ['.001','.002','.003'], 'v')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, wSeq_5 = getData_palm(dir_5, jobName, 'M01', ['.001','.002','.003'], 'w')
f_seq_5, PSD_u_seq_5_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, uSeq_5, 20480)
f_seq_5, PSD_v_seq_5_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, vSeq_5, 20480)
f_seq_5, PSD_w_seq_5_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, wSeq_5, 20480)
ua_5 = uSeq_5.mean()
var_5 = uSeq_5.var()

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl_3' # based on 'pcr_u12_nbl_1', turn off coriolis force
dir_6 = prjDir + '/' + jobName
tSeq_6, xSeq_6, ySeq_6, zSeq_6, uSeq_6 = getData_palm(dir_6, jobName, 'M01', ['.001','.002','.003'], 'u')
tSeq_6, xSeq_6, ySeq_6, zSeq_6, vSeq_6 = getData_palm(dir_6, jobName, 'M01', ['.001','.002','.003'], 'v')
tSeq_6, xSeq_6, ySeq_6, zSeq_6, wSeq_6 = getData_palm(dir_6, jobName, 'M01', ['.001','.002','.003'], 'w')
f_seq_6, PSD_u_seq_6_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_6, 0, xSeq_6.size, ySeq_6.size, uSeq_6, 20480)
f_seq_6, PSD_v_seq_6_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_6, 0, xSeq_6.size, ySeq_6.size, vSeq_6, 20480)
f_seq_6, PSD_w_seq_6_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_6, 0, xSeq_6.size, ySeq_6.size, wSeq_6, 20480)
ua_6 = uSeq_6.mean()
var_6 = uSeq_6.var()

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_nbl_4' # based on 'pcr_u12_nbl_2', double the ug, vg
dir_7 = prjDir + '/' + jobName
tSeq_7, xSeq_7, ySeq_7, zSeq_7, uSeq_7 = getData_palm(dir_7, jobName, 'M01', ['.000','.001'], 'u')
tSeq_7, xSeq_7, ySeq_7, zSeq_7, vSeq_7 = getData_palm(dir_7, jobName, 'M01', ['.000','.001'], 'v')
tSeq_7, xSeq_7, ySeq_7, zSeq_7, wSeq_7 = getData_palm(dir_7, jobName, 'M01', ['.000','.001'], 'w')
f_seq_7, PSD_u_seq_7_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, uSeq_7, 20480)
f_seq_7, PSD_v_seq_7_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, vSeq_7, 20480)
f_seq_7, PSD_w_seq_7_20 = PSD_palm((144000.0, 146400, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, wSeq_7, 20480)
ua_7 = uSeq_7.mean()
var_7 = uSeq_7.var()


""" check PSD """
fig, ax = plt.subplots(figsize=(5.2,3))
colors = plt.cm.jet(np.linspace(0,1,8))
# colors = ['red','blue','green','salmon','cornflowerblue','lightgreen','k']
dn = 1
# plt.loglog(f_seq_0[0::dn], PSD_u_seq_0_100[0::dn], label='sowfa_gs20', linewidth=1.0, linestyle='--', color=colors[0])


# plt.loglog(f_seq_1[0::dn], PSD_u_seq_1_20[0::dn], label='pcr_u8_nbl', linewidth=1.0, linestyle='-', color=colors[1])
# plt.loglog(f_seq_2[0::dn], PSD_u_seq_2_20[0::dn], label='pcr_u12_cbl', linewidth=1.0, linestyle='--', color=colors[2])
# plt.loglog(f_seq_3[0::dn], PSD_u_seq_3_20[0::dn], label='pcr_u12_nbl', linewidth=1.0, linestyle='--', color=colors[3])
# plt.loglog(f_seq_4[0::dn], PSD_u_seq_4_20[0::dn], label='pcr_u12_nbl_1', linewidth=1.0, linestyle='--', color=colors[4])
# plt.loglog(f_seq_5[0::dn], PSD_u_seq_5_20[0::dn], label='pcr_u12_nbl_2', linewidth=1.0, linestyle='--', color=colors[5])
# plt.loglog(f_seq_6[0::dn], PSD_u_seq_6_20[0::dn], label='pcr_u12_nbl_3', linewidth=1.0, linestyle='--', color=colors[6])
# plt.loglog(f_seq_7[0::dn], PSD_u_seq_7_20[0::dn], label='pcr_u12_nbl_4', linewidth=1.0, linestyle='--', color=colors[7])


### scaled
plt.loglog(f_seq_1[0::dn]/ua_1, PSD_u_seq_1_20[0::dn]/var_1, label='pcr_u8_nbl', linewidth=1.0, linestyle='-', color=colors[1])
plt.loglog(f_seq_2[0::dn]/ua_2, PSD_u_seq_2_20[0::dn]/var_2, label='pcr_u12_cbl', linewidth=1.0, linestyle='--', color=colors[2])
plt.loglog(f_seq_3[0::dn]/ua_3, PSD_u_seq_3_20[0::dn]/var_3, label='pcr_u12_nbl', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_4[0::dn]/ua_4, PSD_u_seq_4_20[0::dn]/var_4, label='pcr_u12_nbl_1', linewidth=1.0, linestyle='--', color=colors[4])
plt.loglog(f_seq_5[0::dn]/ua_5, PSD_u_seq_5_20[0::dn]/var_5, label='pcr_u12_nbl_2', linewidth=1.0, linestyle='--', color=colors[5])
plt.loglog(f_seq_6[0::dn]/ua_6, PSD_u_seq_6_20[0::dn]/var_6, label='pcr_u12_nbl_3', linewidth=1.0, linestyle='--', color=colors[6])
plt.loglog(f_seq_7[0::dn]/ua_7, PSD_u_seq_7_20[0::dn]/var_7, label='pcr_u12_nbl_4', linewidth=1.0, linestyle='--', color=colors[7])

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('S' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e3
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()

# sowfa_gs20 U_mean, uvar = ()
# pcr_u8_nbl U_mean, uvar = (8.4, -0.24), 0.24 (closest to deepwind_NBL)
# pcr_u12_cbl U_mean, uvar = (12.1, -0.45), 0.55 (maria's original case)
# pcr_u12_nbl U_mean, uvar = (13.1, -0.025), 0.57 (based on maria's cbl case, set surface heat flux to 0)
# pcr_u12_nbl_1 U_mean, uvar = (10.0, 1.5), 0.35 (based on pcr_u12_nbl, turn off pressure gradient, turn on coriolis force)
# pcr_u12_nbl_2 U_mean, uvar = (8.3, -0.25), 0.21 (based on pcr_u12_nbl_1, change geostropic wind)
# pcr_u12_nbl_3 U_mean, uvar = (7.4, -0.016), 0.16 (based on pcr_u12_nbl, turn off pressure gradient)
















""" PSD of u,v,w for (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(5.2,3))
# colors = plt.cm.jet(np.linspace(0,1,6))
colors = ['red','blue','green','salmon','cornflowerblue','lightgreen']
dn = 1
plt.loglog(f_seq_0[0::dn], PSD_u_seq_0_100[0::dn], label='u-sowfa', linewidth=1.0, linestyle='-', color=colors[0])
plt.loglog(f_seq_0[0::dn], PSD_v_seq_0_100[0::dn], label='v-sowfa', linewidth=1.0, linestyle='-', color=colors[1])
plt.loglog(f_seq_0[0::dn], PSD_w_seq_0_100[0::dn], label='w-sowfa', linewidth=1.0, linestyle='-', color=colors[2])
plt.loglog(f_seq_1[0::dn], PSD_u_seq_1_100[0::dn], label='u-palm', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_1[0::dn], PSD_v_seq_1_100[0::dn], label='v-palm', linewidth=1.0, linestyle='--', color=colors[4])
plt.loglog(f_seq_1[0::dn], PSD_w_seq_1_100[0::dn], label='w-palm', linewidth=1.0, linestyle='--', color=colors[5])

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('S' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e3
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()


""" PSD of u at various heights for (sowfa vs palm) """
### compute kaimal spectra
kappa = 0.4
z0 = 0.001
## sowfa
# compute uz, uStar
zInd = 4 # should be within the surface layer
zMO_0 = zSeq_0[zInd]
pInd_start = xSeq_0.size*ySeq_0.size*zInd
pInd_end = xSeq_0.size*ySeq_0.size*(zInd+1)
uz_0 = uSeq_0[pInd_start:pInd_end].mean()
uStar_0 = kappa * uz_0 / np.log(zMO_0/z0)
# kaimal spectrum
kaimal_u_100_0 = funcs.kaimal_u(f_seq_0[1:], uz_0, zMO_0, uStar_0)
## palm
# compute uz, uStar
zInd = 4
zMO_1 = zSeq_1[zInd]
uz_1 = uSeq_1[:,zInd,:,:].mean()
uStar_1 = kappa * uz_1 / np.log(zMO_1/z0)



fig, ax = plt.subplots(figsize=(5.2,3))
# colors = plt.cm.jet(np.linspace(0,1,6))
colors = ['red','blue','green','salmon','cornflowerblue','lightgreen']
dn = 1
# plt.loglog(f_seq_0[0::dn], PSD_u_seq_0_20[0::dn], label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors[0])
plt.loglog(f_seq_0[0::dn], PSD_u_seq_0_100[0::dn], label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors[0])
# plt.loglog(f_seq_0[0::dn], PSD_u_seq_0_180[0::dn], label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors[2])
# plt.loglog(f_seq_2[0::dn], PSD_u_seq_2_20[0::dn], label='palm-20m', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_2[0::dn], PSD_u_seq_2_100[0::dn], label='palm-100m', linewidth=1.0, linestyle='-', color=colors[1])
# plt.loglog(f_seq_2[0::dn], PSD_u_seq_2_180[0::dn], label='palm-180m', linewidth=1.0, linestyle='--', color=colors[5])

# kaimal spectra
plt.loglog(f_seq_0[1::dn], kaimal_u_100_0[0::dn], label='kaimal-100m', linewidth=1.0, linestyle=':', color='k')

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('Su' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 0.5 # f_seq.max()
yaxis_min = 1e-4
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su.png'
# plt.savefig('/scratch/projects/deepwind/photo/spectra-statistics' + '/' + saveName, bbox_inches='tight')
plt.show()

""" PSD of v at various heights for (sowfa vs palm) """
### compute kaimal spectra
kappa = 0.4
z0 = 0.001
## sowfa
# compute uz, uStar
zInd = 4 # should be within the surface layer
zMO_0 = zSeq_0[zInd]
pInd_start = xSeq_0.size*ySeq_0.size*zInd
pInd_end = xSeq_0.size*ySeq_0.size*(zInd+1)
uz_0 = uSeq_0[pInd_start:pInd_end].mean()
uStar_0 = kappa * uz_0 / np.log(zMO_0/z0)
# kaimal spectrum
kaimal_v_100_0 = funcs.kaimal_v(f_seq_0[1:], uz_0, zMO_0, uStar_0)
## palm
# compute uz, uStar
zInd = 4
zMO_1 = zSeq_1[zInd]
uz_1 = uSeq_1[:,zInd,:,:].mean()
uStar_1 = kappa * uz_1 / np.log(zMO_1/z0)

fig, ax = plt.subplots(figsize=(5.2,3))
# colors = plt.cm.jet(np.linspace(0,1,6))
colors = ['red','blue','green','salmon','cornflowerblue','lightgreen']
dn = 1
# plt.loglog(f_seq_0[0::dn], PSD_v_seq_0_20[0::dn], label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors[0])
plt.loglog(f_seq_0[0::dn], PSD_v_seq_0_100[0::dn], label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors[0])
# plt.loglog(f_seq_0[0::dn], PSD_v_seq_0_180[0::dn], label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors[2])
# plt.loglog(f_seq_2[0::dn], PSD_v_seq_2_20[0::dn], label='palm-20m', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_2[0::dn], PSD_v_seq_2_100[0::dn], label='palm-100m', linewidth=1.0, linestyle='-', color=colors[1])
# plt.loglog(f_seq_2[0::dn], PSD_v_seq_2_180[0::dn], label='palm-180m', linewidth=1.0, linestyle='--', color=colors[5])

# kaimal spectra
plt.loglog(f_seq_0[1::dn], kaimal_v_100_0[0::dn], label='kaimal-100m', linewidth=1.0, linestyle=':', color='k')

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('Sv' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 0.5 # f_seq.max()
yaxis_min = 1e-4
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Sv.png'
# plt.savefig('/scratch/projects/deepwind/photo/spectra-statistics' + '/' + saveName, bbox_inches='tight')
plt.show()

""" PSD of w at various heights for (sowfa vs palm) """
### compute kaimal spectra
kappa = 0.4
z0 = 0.001
## sowfa
# compute uz, uStar
zInd = 4 # should be within the surface layer
zMO_0 = zSeq_0[zInd]
pInd_start = xSeq_0.size*ySeq_0.size*zInd
pInd_end = xSeq_0.size*ySeq_0.size*(zInd+1)
uz_0 = uSeq_0[pInd_start:pInd_end].mean()
uStar_0 = kappa * uz_0 / np.log(zMO_0/z0)
# kaimal spectrum
kaimal_w_100_0 = funcs.kaimal_w(f_seq_0[1:], uz_0, zMO_0, uStar_0)
## palm
# compute uz, uStar
zInd = 4
zMO_1 = zSeq_1[zInd]
uz_1 = uSeq_1[:,zInd,:,:].mean()
uStar_1 = kappa * uz_1 / np.log(zMO_1/z0)

fig, ax = plt.subplots(figsize=(5.2,3))
# colors = plt.cm.jet(np.linspace(0,1,6))
colors = ['red','blue','green','salmon','cornflowerblue','lightgreen']
dn = 1
# plt.loglog(f_seq_0[0::dn], PSD_w_seq_0_20[0::dn], label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors[0])
plt.loglog(f_seq_0[0::dn], PSD_w_seq_0_100[0::dn], label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors[0])
# plt.loglog(f_seq_0[0::dn], PSD_w_seq_0_180[0::dn], label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors[2])
# plt.loglog(f_seq_2[0::dn], PSD_w_seq_2_20[0::dn], label='palm-20m', linewidth=1.0, linestyle='--', color=colors[3])
plt.loglog(f_seq_2[0::dn], PSD_w_seq_2_100[0::dn], label='palm-100m', linewidth=1.0, linestyle='-', color=colors[1])
# plt.loglog(f_seq_2[0::dn], PSD_w_seq_2_180[0::dn], label='palm-180m', linewidth=1.0, linestyle='--', color=colors[5])

# kaimal spectra
plt.loglog(f_seq_0[1::dn], kaimal_w_100_0[0::dn], label='kaimal-100m', linewidth=1.0, linestyle=':', color='k')

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel('Sw' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 0.5 # f_seq.max()
yaxis_min = 1e-4
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.3), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Sw.png'
# plt.savefig('/scratch/projects/deepwind/photo/spectra-statistics' + '/' + saveName, bbox_inches='tight')
plt.show()
