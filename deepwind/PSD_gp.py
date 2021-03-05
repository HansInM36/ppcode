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

def calc_uStar_sowfa(xNum,yNum,zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    pInd_start = xNum*yNum*zMO_ind
    pInd_end = xNum*yNum*(zMO_ind+1)
    uMO = np.mean(uSeq[pInd_start:pInd_end])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar
def calc_uz_sowfa(xNum,yNum,zInd,uSeq):
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)
    uz = np.mean(uSeq[pInd_start:pInd_end])
    return uz
def calc_uStar_palm(zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    uMO = np.mean(uSeq[:,zMO_ind,:,:])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar
def calc_uz_palm(zInd,uSeq):
    uz = np.mean(uSeq[:,zInd,:,:])
    return uz


segNum = 4096

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_0.0001_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10_refined'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_1, PSD_u_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_v_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_w_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs10_0.01_refined'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, vSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, wSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_2, PSD_u_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_v_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_w_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)



prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_0.0001_main'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3 = getData_palm(dir_3, jobName, 'M03', ['.001'], 'u')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, vSeq_3 = getData_palm(dir_3, jobName, 'M03', ['.001'], 'v')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, wSeq_3 = getData_palm(dir_3, jobName, 'M03', ['.001'], 'w')
f_seq_3, PSD_u_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_v_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_w_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'u')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, vSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'v')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, wSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'w')
f_seq_4, PSD_u_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_u_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_u_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_v_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_v_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_v_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_w_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)
f_seq_4, PSD_w_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)
f_seq_4, PSD_w_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_0.01_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.000'], 'u')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, vSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.000'], 'v')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, wSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.000'], 'w')
f_seq_5, PSD_u_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_v_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_w_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)



""" 3*3 plots of scaled PSD (h) """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

# compute uz, uStar in SOWFA
uStar_0 = calc_uStar_sowfa(xSeq_0.size,ySeq_0.size,zSeq_0,uSeq_0,kappa=0.4,z0=0.0001)
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,0,uSeq_0)
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,uSeq_0)
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,8,uSeq_0)
# compute uz, uStar in PALM
uStar_3 = calc_uStar_palm(zSeq_3,uSeq_3,0,kappa=0.4,z0=0.0001)
uz_3_20 = calc_uz_palm(0,uSeq_3)
uz_3_100 = calc_uz_palm(4,uSeq_3)
uz_3_180 = calc_uz_palm(8,uSeq_3)

axs[0,0].loglog(f_seq_0*20/uz_0_20, PSD_u_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,0].loglog(f_seq_0*100/uz_0_100, PSD_u_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,0].loglog(f_seq_0*180/uz_0_180, PSD_u_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,0].loglog(f_seq_3*20/uz_3_20, PSD_u_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,0].loglog(f_seq_3*100/uz_3_100, PSD_u_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,0].loglog(f_seq_3*180/uz_3_180, PSD_u_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,0].set_xlim(1e-3, 1e2); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(1e-10, 1e2)
axs[0,0].set_ylabel(r"$\mathrm{fS_u/u^2_*}$", fontsize=12)
axs[0,0].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[1,0].loglog(f_seq_0*20/uz_0_20, PSD_v_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,0].loglog(f_seq_0*100/uz_0_100, PSD_v_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,0].loglog(f_seq_0*180/uz_0_180, PSD_v_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,0].loglog(f_seq_3*20/uz_3_20, PSD_v_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,0].loglog(f_seq_3*100/uz_3_100, PSD_v_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,0].loglog(f_seq_3*180/uz_3_180, PSD_v_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,0].set_xlim(1e-3, 1e2); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(1e-10, 1e2)
axs[1,0].set_ylabel(r"$\mathrm{fS_v/u^2_*}$", fontsize=12)
axs[1,0].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[2,0].loglog(f_seq_0*20/uz_0_20, PSD_w_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,0].loglog(f_seq_0*100/uz_0_100, PSD_w_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,0].loglog(f_seq_0*180/uz_0_180, PSD_w_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,0].loglog(f_seq_3*20/uz_3_20, PSD_w_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,0].loglog(f_seq_3*100/uz_3_100, PSD_w_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,0].loglog(f_seq_3*180/uz_3_180, PSD_w_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,0].set_xlim(1e-3, 1e2)
axs[2,0].set_ylim(1e-10, 1e2)
axs[2,0].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,0].set_ylabel(r"$\mathrm{fS_w/u^2_*}$", fontsize=12)
axs[2,0].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes

# compute uz, uStar in SOWFA
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
# compute uz, uStar in PALM
uStar_4 = calc_uStar_palm(zSeq_4,uSeq_4,0,kappa=0.4,z0=0.001)
uz_4_20 = calc_uz_palm(0,uSeq_4)
uz_4_100 = calc_uz_palm(4,uSeq_4)
uz_4_180 = calc_uz_palm(8,uSeq_4)

axs[0,1].loglog(f_seq_1*20/uz_1_20, PSD_u_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,1].loglog(f_seq_1*100/uz_1_100, PSD_u_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,1].loglog(f_seq_1*180/uz_1_180, PSD_u_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,1].loglog(f_seq_4*20/uz_4_20, PSD_u_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,1].loglog(f_seq_4*100/uz_4_100, PSD_u_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,1].loglog(f_seq_4*180/uz_4_180, PSD_u_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,1].set_xlim(1e-3, 1e2); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(1e-10, 1e2); axs[0,1].set_yticklabels([])
axs[0,1].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[1,1].loglog(f_seq_1*20/uz_1_20, PSD_v_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,1].loglog(f_seq_1*100/uz_1_100, PSD_v_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,1].loglog(f_seq_1*180/uz_1_180, PSD_v_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,1].loglog(f_seq_4*20/uz_4_20, PSD_v_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,1].loglog(f_seq_4*100/uz_4_100, PSD_v_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,1].loglog(f_seq_4*180/uz_4_180, PSD_v_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,1].set_xlim(1e-3, 1e2); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(1e-10, 1e2); axs[1,1].set_yticklabels([])
axs[1,1].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[2,1].loglog(f_seq_1*20/uz_1_20, PSD_w_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,1].loglog(f_seq_1*100/uz_1_100, PSD_w_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,1].loglog(f_seq_1*180/uz_1_180, PSD_w_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,1].loglog(f_seq_4*20/uz_4_20, PSD_w_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,1].loglog(f_seq_4*100/uz_4_100, PSD_w_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,1].loglog(f_seq_4*180/uz_4_180, PSD_w_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,1].set_xlim(1e-3, 1e2)
axs[2,1].set_ylim(1e-10, 1e2); axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,1].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes

# compute uz, uStar in SOWFA
uStar_2 = calc_uStar_sowfa(xSeq_2.size,ySeq_2.size,zSeq_2,uSeq_2,kappa=0.4,z0=0.01)
uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,0,uSeq_2)
uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,uSeq_2)
uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,8,uSeq_2)
# compute uz, uStar in PALM
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.01)
uz_5_20 = calc_uz_palm(0,uSeq_5)
uz_5_100 = calc_uz_palm(4,uSeq_5)
uz_5_180 = calc_uz_palm(8,uSeq_5)

axs[0,2].loglog(f_seq_2*20/uz_2_20, PSD_u_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,2].loglog(f_seq_2*100/uz_2_100, PSD_u_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,2].loglog(f_seq_2*180/uz_2_180, PSD_u_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,2].loglog(f_seq_5*20/uz_5_20, PSD_u_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,2].loglog(f_seq_5*100/uz_5_100, PSD_u_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,2].loglog(f_seq_5*180/uz_5_180, PSD_u_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,2].set_xlim(1e-3, 1e2); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(1e-10, 1e2); axs[0,2].set_yticklabels([])
axs[0,2].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[1,2].loglog(f_seq_2*20/uz_2_20, PSD_v_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,2].loglog(f_seq_2*100/uz_2_100, PSD_v_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,2].loglog(f_seq_2*180/uz_2_180, PSD_v_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,2].loglog(f_seq_5*20/uz_5_20, PSD_v_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,2].loglog(f_seq_5*100/uz_5_100, PSD_v_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,2].loglog(f_seq_5*180/uz_5_180, PSD_v_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,2].set_xlim(1e-3, 1e2); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(1e-10, 1e2); axs[1,2].set_yticklabels([])
axs[1,2].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[2,2].loglog(f_seq_2*20/uz_2_20, PSD_w_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-20m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,2].loglog(f_seq_2*100/uz_2_100, PSD_w_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-100m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,2].loglog(f_seq_2*180/uz_2_180, PSD_w_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-180m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,2].loglog(f_seq_5*20/uz_5_20, PSD_w_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-20m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,2].loglog(f_seq_5*100/uz_5_100, PSD_w_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-100m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,2].loglog(f_seq_5*180/uz_5_180, PSD_w_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-180m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,2].set_xlim(1e-3, 1e2)
axs[2,2].set_ylim(1e-10, 1e2); axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,2].text(5e-3, 1e-8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,3,1,4,2,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=3, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/PSD'
saveName = 'PSD_scaled_gp_h.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()



""" 3*3 plots of scaled PSD (z0) """
segNum = 4096

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

# compute uz, uStar in SOWFA
uStar_0 = calc_uStar_sowfa(xSeq_0.size,ySeq_0.size,zSeq_0,uSeq_0,kappa=0.4,z0=0.0001)
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uStar_2 = calc_uStar_sowfa(xSeq_2.size,ySeq_2.size,zSeq_2,uSeq_2,kappa=0.4,z0=0.01)
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,0,uSeq_0)
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,uSeq_0)
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,8,uSeq_0)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,0,uSeq_2)
uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,uSeq_2)
uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,8,uSeq_2)

# compute uz, uStar in PALM
uStar_3 = calc_uStar_palm(zSeq_3,uSeq_3,0,kappa=0.4,z0=0.0001)
uStar_4 = calc_uStar_palm(zSeq_4,uSeq_4,0,kappa=0.4,z0=0.001)
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.01)
uz_3_20 = calc_uz_palm(0,uSeq_3)
uz_3_100 = calc_uz_palm(4,uSeq_3)
uz_3_180 = calc_uz_palm(8,uSeq_3)
uz_4_20 = calc_uz_palm(0,uSeq_4)
uz_4_100 = calc_uz_palm(4,uSeq_4)
uz_4_180 = calc_uz_palm(8,uSeq_4)
uz_5_20 = calc_uz_palm(0,uSeq_5)
uz_5_100 = calc_uz_palm(4,uSeq_5)
uz_5_180 = calc_uz_palm(8,uSeq_5)

# kaimal spectrum
f_seq_kaimal = np.logspace(-4,2,2049)
# f_seq_kaimal = np.linspace(1e-4,1e0,2049)
kaimal_u_20_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_u_100_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_u_180_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
kaimal_v_20_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_v_100_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_v_180_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
kaimal_w_20_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_w_100_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_w_180_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
f_seq_BP = np.logspace(-4,2,2049)
# f_seq_BP = np.linspace(1e-4,1e0,2049)
BP_w_20_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_20, zSeq_1[0], uStar_1)
BP_w_100_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_100, zSeq_1[4], uStar_1)
BP_w_180_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_180, zSeq_1[8], uStar_1)

axs[0,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_u_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,0].loglog(f_seq_0*20/uz_0_20, PSD_u_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,0].loglog(f_seq_1*20/uz_1_20, PSD_u_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,0].loglog(f_seq_2*20/uz_2_20, PSD_u_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,0].loglog(f_seq_3*20/uz_3_20, PSD_u_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,0].loglog(f_seq_4*20/uz_4_20, PSD_u_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,0].loglog(f_seq_5*20/uz_5_20, PSD_u_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,0].set_xlim(1e-2, 1e1); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(1e-4, 1e1)
axs[0,0].set_ylabel(r"$\mathrm{fS_u/u^2_*}$", fontsize=12)
axs[0,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[0,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_u_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,1].loglog(f_seq_0*100/uz_0_100, PSD_u_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,1].loglog(f_seq_1*100/uz_1_100, PSD_u_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,1].loglog(f_seq_2*100/uz_2_100, PSD_u_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,1].loglog(f_seq_3*100/uz_3_100, PSD_u_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,1].loglog(f_seq_4*100/uz_4_100, PSD_u_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,1].loglog(f_seq_5*100/uz_5_100, PSD_u_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,1].set_xlim(1e-2, 1e1); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(1e-4, 1e1); axs[0,1].set_yticklabels([])
axs[0,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,1].transAxes

axs[0,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_u_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,2].loglog(f_seq_0*180/uz_0_180, PSD_u_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,2].loglog(f_seq_1*180/uz_1_180, PSD_u_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,2].loglog(f_seq_2*180/uz_2_180, PSD_u_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,2].loglog(f_seq_3*180/uz_3_180, PSD_u_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,2].loglog(f_seq_4*180/uz_4_180, PSD_u_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,2].loglog(f_seq_5*180/uz_5_180, PSD_u_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,2].set_xlim(1e-2, 1e1); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(1e-4, 1e1); axs[0,2].set_yticklabels([])
axs[0,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,2].transAxes

axs[1,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_v_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,0].loglog(f_seq_0*20/uz_0_20, PSD_v_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,0].loglog(f_seq_1*20/uz_1_20, PSD_v_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,0].loglog(f_seq_2*20/uz_2_20, PSD_v_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,0].loglog(f_seq_3*20/uz_3_20, PSD_v_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,0].loglog(f_seq_4*20/uz_4_20, PSD_v_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,0].loglog(f_seq_5*20/uz_5_20, PSD_v_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,0].set_xlim(1e-2, 1e1); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(1e-4, 1e1)
axs[1,0].set_ylabel(r"$\mathrm{fS_v/u^2_*}$", fontsize=12)
axs[1,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[1,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_v_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,1].loglog(f_seq_0*100/uz_0_100, PSD_v_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,1].loglog(f_seq_1*100/uz_1_100, PSD_v_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,1].loglog(f_seq_2*100/uz_2_100, PSD_v_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,1].loglog(f_seq_3*100/uz_3_100, PSD_v_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,1].loglog(f_seq_4*100/uz_4_100, PSD_v_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,1].loglog(f_seq_5*100/uz_5_100, PSD_v_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,1].set_xlim(1e-2, 1e1); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(1e-4, 1e1); axs[1,1].set_yticklabels([])
axs[1,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[1,1].transAxes

axs[1,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_v_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,2].loglog(f_seq_0*180/uz_0_180, PSD_v_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,2].loglog(f_seq_1*180/uz_1_180, PSD_v_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,2].loglog(f_seq_2*180/uz_2_180, PSD_v_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,2].loglog(f_seq_3*180/uz_3_180, PSD_v_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,2].loglog(f_seq_4*180/uz_4_180, PSD_v_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,2].loglog(f_seq_5*180/uz_5_180, PSD_v_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,2].set_xlim(1e-2, 1e1); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(1e-4, 1e1); axs[1,2].set_yticklabels([])
axs[1,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[1,2].transAxes

axs[2,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_w_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,0].loglog(f_seq_BP*20/uz_1_20, BP_w_20_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,0].loglog(f_seq_0*20/uz_0_20, PSD_w_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,0].loglog(f_seq_1*20/uz_1_20, PSD_w_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,0].loglog(f_seq_2*20/uz_2_20, PSD_w_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,0].loglog(f_seq_3*20/uz_3_20, PSD_w_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,0].loglog(f_seq_4*20/uz_4_20, PSD_w_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,0].loglog(f_seq_5*20/uz_5_20, PSD_w_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,0].set_xlim(1e-2, 1e1)
axs[2,0].set_ylim(1e-4, 1e1)
axs[2,0].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,0].set_ylabel(r"$\mathrm{fS_w/u^2_*}$", fontsize=12)
axs[2,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[2,0].transAxes

axs[2,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_w_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,1].loglog(f_seq_BP*100/uz_1_100, BP_w_100_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,1].loglog(f_seq_0*100/uz_0_100, PSD_w_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,1].loglog(f_seq_1*100/uz_1_100, PSD_w_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,1].loglog(f_seq_2*100/uz_2_100, PSD_w_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,1].loglog(f_seq_3*100/uz_3_100, PSD_w_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,1].loglog(f_seq_4*100/uz_4_100, PSD_w_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,1].loglog(f_seq_5*100/uz_5_100, PSD_w_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,1].set_xlim(1e-2, 1e1)
axs[2,1].set_ylim(1e-4, 1e1); axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[2,1].transAxes

axs[2,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_w_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,2].loglog(f_seq_BP*180/uz_1_180, BP_w_180_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,2].loglog(f_seq_0*180/uz_0_180, PSD_w_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,2].loglog(f_seq_1*180/uz_1_180, PSD_w_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,2].loglog(f_seq_2*180/uz_2_180, PSD_w_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,2].loglog(f_seq_3*180/uz_3_180, PSD_w_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,2].loglog(f_seq_4*180/uz_4_180, PSD_w_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,2].loglog(f_seq_5*180/uz_5_180, PSD_w_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,2].set_xlim(1e-2, 1e1)
axs[2,2].set_ylim(1e-4, 1e1); axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[2,2].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[2,0].get_legend_handles_labels()
lgdord = [2,5,3,6,4,7,0,1]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=4, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/PSD'
saveName = 'PSD_scaled_gp_z0_zoomin_mdf.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()


""" 3*3 plots of PSD """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

axs[0,0].loglog(f_seq_0, PSD_u_seq_0_20, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,0].loglog(f_seq_1, PSD_u_seq_1_20, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,0].loglog(f_seq_2, PSD_u_seq_2_20, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,0].loglog(f_seq_3, PSD_u_seq_3_20, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,0].loglog(f_seq_4, PSD_u_seq_4_20, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,0].loglog(f_seq_5, PSD_u_seq_5_20, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,0].set_xlim(1e-3, 1e0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(1e-10, 1e2)
axs[0,0].set_ylabel(r"$\mathrm{fS_u/u^2_*}$", fontsize=12)
axs[0,0].text(5e-3, 1e-8, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[0,1].loglog(f_seq_0, PSD_u_seq_0_100, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,1].loglog(f_seq_1, PSD_u_seq_1_100, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,1].loglog(f_seq_2, PSD_u_seq_2_100, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,1].loglog(f_seq_3, PSD_u_seq_3_100, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,1].loglog(f_seq_4, PSD_u_seq_4_100, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,1].loglog(f_seq_5, PSD_u_seq_5_100, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,1].set_xlim(1e-3, 1e0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(1e-10, 1e2); axs[0,1].set_yticklabels([])
axs[0,1].text(5e-3, 1e-8, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,1].transAxes

axs[0,2].loglog(f_seq_0, PSD_u_seq_0_180, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,2].loglog(f_seq_1, PSD_u_seq_1_180, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,2].loglog(f_seq_2, PSD_u_seq_2_180, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,2].loglog(f_seq_3, PSD_u_seq_3_180, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,2].loglog(f_seq_4, PSD_u_seq_4_180, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,2].loglog(f_seq_5, PSD_u_seq_5_180, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,2].set_xlim(1e-3, 1e0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(1e-10, 1e2); axs[0,2].set_yticklabels([])
axs[0,2].text(5e-3, 1e-8, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,2].transAxes

axs[1,0].loglog(f_seq_0, PSD_v_seq_0_20, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,0].loglog(f_seq_1, PSD_v_seq_1_20, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,0].loglog(f_seq_2, PSD_v_seq_2_20, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,0].loglog(f_seq_3, PSD_v_seq_3_20, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,0].loglog(f_seq_4, PSD_v_seq_4_20, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,0].loglog(f_seq_5, PSD_v_seq_5_20, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,0].set_xlim(1e-3, 1e0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(1e-10, 1e2)
axs[1,0].set_ylabel(r"$\mathrm{fS_v/u^2_*}$", fontsize=12)
axs[1,0].text(5e-3, 1e-8, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

axs[1,1].loglog(f_seq_0, PSD_v_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,1].loglog(f_seq_1, PSD_v_seq_1_100, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,1].loglog(f_seq_2, PSD_v_seq_2_100, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,1].loglog(f_seq_3, PSD_v_seq_3_100, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,1].loglog(f_seq_4, PSD_v_seq_4_100, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,1].loglog(f_seq_5, PSD_v_seq_5_100, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,1].set_xlim(1e-3, 1e0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(1e-10, 1e2); axs[1,1].set_yticklabels([])
axs[1,1].text(5e-3, 1e-8, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[1,1].transAxes

axs[1,2].loglog(f_seq_0, PSD_v_seq_0_180, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,2].loglog(f_seq_1, PSD_v_seq_1_180, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,2].loglog(f_seq_2, PSD_v_seq_2_180, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,2].loglog(f_seq_3, PSD_v_seq_3_180, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,2].loglog(f_seq_4, PSD_v_seq_4_180, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,2].loglog(f_seq_5, PSD_v_seq_5_180, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,2].set_xlim(1e-3, 1e0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(1e-10, 1e2); axs[1,2].set_yticklabels([])
axs[1,2].text(5e-3, 1e-8, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[1,2].transAxes

axs[2,0].loglog(f_seq_0, PSD_w_seq_0_20, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,0].loglog(f_seq_1, PSD_w_seq_1_20, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,0].loglog(f_seq_2, PSD_w_seq_2_20, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,0].loglog(f_seq_3, PSD_w_seq_3_20, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,0].loglog(f_seq_4, PSD_w_seq_4_20, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,0].loglog(f_seq_5, PSD_w_seq_5_20, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,0].set_xlim(1e-3, 1e0)
axs[2,0].set_ylim(1e-10, 1e2)
axs[2,0].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,0].set_ylabel(r"$\mathrm{fS_w/u^2_*}$", fontsize=12)
axs[2,0].text(5e-3, 1e-8, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[2,0].transAxes

axs[2,1].loglog(f_seq_0, PSD_w_seq_0_100, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,1].loglog(f_seq_1, PSD_w_seq_1_100, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,1].loglog(f_seq_2, PSD_w_seq_2_100, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,1].loglog(f_seq_3, PSD_w_seq_3_100, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,1].loglog(f_seq_4, PSD_w_seq_4_100, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,1].loglog(f_seq_5, PSD_w_seq_5_100, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,1].set_xlim(1e-3, 1e0)
axs[2,1].set_ylim(1e-10, 1e2); axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,1].text(5e-3, 1e-8, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[2,1].transAxes

axs[2,2].loglog(f_seq_0, PSD_w_seq_0_180, label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,2].loglog(f_seq_1, PSD_w_seq_1_180, label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,2].loglog(f_seq_2, PSD_w_seq_2_180, label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,2].loglog(f_seq_3, PSD_w_seq_3_180, label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,2].loglog(f_seq_4, PSD_w_seq_4_180, label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,2].loglog(f_seq_5, PSD_w_seq_5_180, label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,2].set_xlim(1e-3, 1e0)
axs[2,2].set_ylim(1e-10, 1e2); axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=12)
axs[2,2].text(5e-3, 1e-8, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[2,2].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,3,1,4,2,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.94), ncol=3, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/PSD'
saveName = 'PSD_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
