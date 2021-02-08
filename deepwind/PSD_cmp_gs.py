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
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)

# f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
# f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
# f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
# f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
# f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
# f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
# f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
# f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
# f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)

segNum = int(0.4*tSeq_0.size)
f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)


# prjDir = '/scratch/sowfadata/JOBS'
# prjName = 'deepwind'
# jobName_8 = 'gs20_ck0.1'
# ppDir_8 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_8
# tSeq_8, xSeq_8, ySeq_8, zSeq_8, uSeq_8, coors_8 = getData_sowfa(ppDir_8, 'prbg0', ((0,0,0),30.0), 'U', 0)
# tSeq_8, xSeq_8, ySeq_8, zSeq_8, vSeq_8, coors_8 = getData_sowfa(ppDir_8, 'prbg0', ((0,0,0),30.0), 'U', 1)
# tSeq_8, xSeq_8, ySeq_8, zSeq_8, wSeq_8, coors_8 = getData_sowfa(ppDir_8, 'prbg0', ((0,0,0),30.0), 'U', 2)
# segNum = int(0.4*tSeq_8.size)
# f_seq_8, PSD_u_seq_8_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 0, xSeq_8.size, ySeq_8.size, uSeq_8, segNum)
# f_seq_8, PSD_u_seq_8_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 4, xSeq_8.size, ySeq_8.size, uSeq_8, segNum)
# f_seq_8, PSD_u_seq_8_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 8, xSeq_8.size, ySeq_8.size, uSeq_8, segNum)
# f_seq_8, PSD_v_seq_8_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 0, xSeq_8.size, ySeq_8.size, vSeq_8, segNum)
# f_seq_8, PSD_v_seq_8_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 4, xSeq_8.size, ySeq_8.size, vSeq_8, segNum)
# f_seq_8, PSD_v_seq_8_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 8, xSeq_8.size, ySeq_8.size, vSeq_8, segNum)
# f_seq_8, PSD_w_seq_8_20 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 0, xSeq_8.size, ySeq_8.size, wSeq_8, segNum)
# f_seq_8, PSD_w_seq_8_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 4, xSeq_8.size, ySeq_8.size, wSeq_8, segNum)
# f_seq_8, PSD_w_seq_8_180 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_8, 8, xSeq_8.size, ySeq_8.size, wSeq_8, segNum)
#



prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 2)

segNum = int(0.4*tSeq_1.size)
f_seq_1, PSD_u_seq_1_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_v_seq_1_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_w_seq_1_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 0, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 8, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs20'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, vSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, wSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 2)
segNum = int(0.4*tSeq_2.size)
f_seq_2, PSD_u_seq_2_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_v_seq_2_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_w_seq_2_20 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 0, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_180 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 8, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_3 = 'gs40'
ppDir_3 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_3
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3, coors_3 = getData_sowfa(ppDir_3, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_3, xSeq_3, ySeq_3, zSeq_3, vSeq_3, coors_3 = getData_sowfa(ppDir_3, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_3, xSeq_3, ySeq_3, zSeq_3, wSeq_3, coors_3 = getData_sowfa(ppDir_3, 'prbg0', ((0,0,0),30.0), 'U', 2)
segNum = int(0.4*tSeq_3.size)
f_seq_3, PSD_u_seq_3_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 4, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 8, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_v_seq_3_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 4, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 8, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_w_seq_3_20 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 0, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 4, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_180 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 8, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)



prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'u')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, vSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'v')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, wSeq_4 = getData_palm(dir_4, jobName, 'M03', ['.001'], 'w')
segNum = int(0.4*tSeq_4.size)
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
jobName  = 'deepwind_NBL_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.006','.007'], 'u')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, vSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.006','.007'], 'v')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, wSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.006','.007'], 'w')
segNum = int(0.4*tSeq_5.size)
f_seq_5, PSD_u_seq_5_20 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_100 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 4, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_180 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 8, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_v_seq_5_20 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_100 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 4, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_180 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 8, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_w_seq_5_20 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 0, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_100 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 4, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_180 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 8, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs20_main'
dir_6 = prjDir + '/' + jobName
tSeq_6, xSeq_6, ySeq_6, zSeq_6, uSeq_6 = getData_palm(dir_6, jobName, 'M03', ['.001'], 'u')
tSeq_6, xSeq_6, ySeq_6, zSeq_6, vSeq_6 = getData_palm(dir_6, jobName, 'M03', ['.001'], 'v')
tSeq_6, xSeq_6, ySeq_6, zSeq_6, wSeq_6 = getData_palm(dir_6, jobName, 'M03', ['.001'], 'w')
segNum = int(0.4*tSeq_6.size)
f_seq_6, PSD_u_seq_6_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 0, xSeq_6.size, ySeq_6.size, uSeq_6, segNum)
f_seq_6, PSD_u_seq_6_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 4, xSeq_6.size, ySeq_6.size, uSeq_6, segNum)
f_seq_6, PSD_u_seq_6_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 8, xSeq_6.size, ySeq_6.size, uSeq_6, segNum)
f_seq_6, PSD_v_seq_6_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 0, xSeq_6.size, ySeq_6.size, vSeq_6, segNum)
f_seq_6, PSD_v_seq_6_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 4, xSeq_6.size, ySeq_6.size, vSeq_6, segNum)
f_seq_6, PSD_v_seq_6_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 8, xSeq_6.size, ySeq_6.size, vSeq_6, segNum)
f_seq_6, PSD_w_seq_6_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 0, xSeq_6.size, ySeq_6.size, wSeq_6, segNum)
f_seq_6, PSD_w_seq_6_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 4, xSeq_6.size, ySeq_6.size, wSeq_6, segNum)
f_seq_6, PSD_w_seq_6_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 8, xSeq_6.size, ySeq_6.size, wSeq_6, segNum)



prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs40_main'
dir_7 = prjDir + '/' + jobName
tSeq_7, xSeq_7, ySeq_7, zSeq_7, uSeq_7 = getData_palm(dir_7, jobName, 'M03', ['.004'], 'u')
tSeq_7, xSeq_7, ySeq_7, zSeq_7, vSeq_7 = getData_palm(dir_7, jobName, 'M03', ['.004'], 'v')
tSeq_7, xSeq_7, ySeq_7, zSeq_7, wSeq_7 = getData_palm(dir_7, jobName, 'M03', ['.004'], 'w')
segNum = int(0.4*tSeq_7.size)
f_seq_7, PSD_u_seq_7_20 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, uSeq_7, segNum)
f_seq_7, PSD_u_seq_7_100 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 2, xSeq_7.size, ySeq_7.size, uSeq_7, segNum)
f_seq_7, PSD_u_seq_7_180 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 4, xSeq_7.size, ySeq_7.size, uSeq_7, segNum)
f_seq_7, PSD_v_seq_7_20 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, vSeq_7, segNum)
f_seq_7, PSD_v_seq_7_100 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 2, xSeq_7.size, ySeq_7.size, vSeq_7, segNum)
f_seq_7, PSD_v_seq_7_180 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 4, xSeq_7.size, ySeq_7.size, vSeq_7, segNum)
f_seq_7, PSD_w_seq_7_20 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 0, xSeq_7.size, ySeq_7.size, wSeq_7, segNum)
f_seq_7, PSD_w_seq_7_100 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 2, xSeq_7.size, ySeq_7.size, wSeq_7, segNum)
f_seq_7, PSD_w_seq_7_180 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 4, xSeq_7.size, ySeq_7.size, wSeq_7, segNum)





""" Su cmp gs """
# kaimal spectrum
uStar_0 = calc_uStar_sowfa(xSeq_0.size,ySeq_0.size,zSeq_0,uSeq_0,kappa=0.4,z0=0.001)
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,uSeq_0)
fseq = np.copy(f_seq_0); fseq[0] = 1e-3
# kaimal_u_100_0 = funcs.kaimal_u(fseq, uz_0_100, zSeq_0[4], uStar_0)

# plot
fig, ax = plt.subplots(figsize=(5,5))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_0, PSD_u_seq_0_100, label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1, PSD_u_seq_1_100, label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_2, PSD_u_seq_2_100, label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
plt.loglog(f_seq_3, PSD_u_seq_3_100, label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
plt.loglog(f_seq_4, PSD_u_seq_4_100, label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
plt.loglog(f_seq_5, PSD_u_seq_5_100, label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
plt.loglog(f_seq_6, PSD_u_seq_6_100, label='palm-gs20', linewidth=1.0, linestyle='-', color=colors_palm[2])
plt.loglog(f_seq_7, PSD_u_seq_7_100, label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])

## plot kaimal
# plt.loglog(fseq, kaimal_u_100_0, label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e0*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1 # f_seq.max()
yaxis_min = 1e-12
yaxis_max = 1e4
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.05,0.32), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_cmp_gs.png'
# plt.savefig('/scratch/projects/deepwind/photo/PSD' + '/' + saveName, bbox_inches='tight')
plt.show()


""" Su cmp scaled gs """
## sowfa
# compute uz, uStar
uStar_0 = calc_uStar_sowfa(xSeq_0.size,ySeq_0.size,zSeq_0,uSeq_0,kappa=0.4,z0=0.001)
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,uSeq_0)
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uStar_2 = calc_uStar_sowfa(xSeq_2.size,ySeq_2.size,zSeq_2,uSeq_2,kappa=0.4,z0=0.001)
uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,uSeq_2)
uStar_3 = calc_uStar_sowfa(xSeq_3.size,ySeq_3.size,zSeq_3,uSeq_3,kappa=0.4,z0=0.001)
uz_3_100 = calc_uz_sowfa(xSeq_3.size,ySeq_3.size,4,uSeq_3)

## palm
# compute uz, uStar
uStar_4 = calc_uStar_palm(zSeq_4,uSeq_4,0,kappa=0.4,z0=0.001)
uz_4_100 = calc_uz_palm(4,uSeq_4)
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_100 = calc_uz_palm(4,uSeq_5)
uStar_6 = calc_uStar_palm(zSeq_6,uSeq_6,0,kappa=0.4,z0=0.001)
uz_6_100 = calc_uz_palm(4,uSeq_6)
uStar_7 = calc_uStar_palm(zSeq_7,uSeq_7,0,kappa=0.4,z0=0.001)
uz_7_100 = calc_uz_palm(2,uSeq_7)

# # kaimal spectrum
# kaimal_u_100_0 = funcs.kaimal_u(f_seq_0[1:], uz_0_100, zSeq_0[4], uStar_0)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_0*100/uz_0_100, PSD_u_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*100/uz_1_100, PSD_u_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_2*100/uz_2_100, PSD_u_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
plt.loglog(f_seq_3*100/uz_3_100, PSD_u_seq_3_100*f_seq_3/np.power(uStar_3,2), label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
plt.loglog(f_seq_4*100/uz_4_100, PSD_u_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
plt.loglog(f_seq_5*100/uz_5_100, PSD_u_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
plt.loglog(f_seq_6*100/uz_6_100, PSD_u_seq_6_100*f_seq_6/np.power(uStar_6,2), label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[2])
plt.loglog(f_seq_7*100/uz_7_100, PSD_u_seq_7_100*f_seq_7/np.power(uStar_7,2), label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])

# # plot kaimal
# plt.loglog(f_seq_0[1:]*100/uz_0_100, kaimal_u_100_0*f_seq_0[1:]/np.power(uStar_0,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_u/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 50 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e4
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_scaled_cmp_gs.png'
# plt.savefig('/scratch/projects/deepwind/photo/PSD' + '/' + saveName, bbox_inches='tight')
plt.show()











""" Su cmp z0 - 100m """
# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9, PSD_u_seq_9_100, label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1, PSD_u_seq_1_100, label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8, PSD_u_seq_8_100, label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5, PSD_u_seq_5_100, label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# plot -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1e0 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_cmp_z0_h100.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Su scaled cmp z0 - 100m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_100 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,4,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_100 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,4,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_100 = calc_uz_palm(4,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*100/uz_9_100, PSD_u_seq_9_100*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*100/uz_1_100, PSD_u_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*100/uz_8_100, PSD_u_seq_8_100*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*100/uz_5_100, PSD_u_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_u_100_1 = funcs.kaimal_u(f_seq_1[1:], uz_1_100, zSeq_1[4], uStar_1)
plt.loglog(f_seq_1[1:]*100/uz_1_100, kaimal_u_100_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_u/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_scaled_cmp_z0_h100.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()


""" Su cmp z0 - 20m """
# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9, PSD_u_seq_9_20, label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1, PSD_u_seq_1_20, label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8, PSD_u_seq_8_20, label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5, PSD_u_seq_5_20, label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# plot -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1e0 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_cmp_z0_h20.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Su scaled cmp z0 - 20m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_20 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,0,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_20 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,0,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_20 = calc_uz_palm(0,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*20/uz_9_20, PSD_u_seq_9_20*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*20/uz_1_20, PSD_u_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*20/uz_8_20, PSD_u_seq_8_20*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*20/uz_5_20, PSD_u_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_u_20_1 = funcs.kaimal_u(f_seq_1[1:], uz_1_20, zSeq_1[0], uStar_1)
plt.loglog(f_seq_1[1:]*20/uz_1_20, kaimal_u_20_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_u/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_scaled_cmp_z0_h20.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()


""" Su cmp z0 - 180m """
# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9, PSD_u_seq_9_180, label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1, PSD_u_seq_1_180, label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8, PSD_u_seq_8_180, label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5, PSD_u_seq_5_180, label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# plot -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 1e0 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_cmp_z0_h180.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Su scaled cmp z0 - 180m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_180 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,8,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_180 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,8,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_180 = calc_uz_palm(8,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*180/uz_9_180, PSD_u_seq_9_180*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*180/uz_1_180, PSD_u_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*180/uz_8_180, PSD_u_seq_8_180*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*180/uz_5_180, PSD_u_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_u_180_1 = funcs.kaimal_u(f_seq_1[1:], uz_1_180, zSeq_1[8], uStar_1)
plt.loglog(f_seq_1[1:]*180/uz_1_180, kaimal_u_180_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_u/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su_f_scaled_cmp_z0_h180.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()








""" Sv scaled cmp z0 - 100m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_100 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,4,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_100 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,4,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_100 = calc_uz_palm(4,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*100/uz_9_100, PSD_v_seq_9_100*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*100/uz_1_100, PSD_v_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*100/uz_8_100, PSD_v_seq_8_100*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*100/uz_5_100, PSD_v_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_v_100_1 = funcs.kaimal_v(f_seq_1[1:], uz_1_100, zSeq_1[4], uStar_1)
plt.loglog(f_seq_1[1:]*100/uz_1_100, kaimal_v_100_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_v/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sv_f_scaled_cmp_z0_h100.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Sv scaled cmp z0 - 20m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_20 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,0,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_20 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,0,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_20 = calc_uz_palm(0,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*20/uz_9_20, PSD_v_seq_9_20*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*20/uz_1_20, PSD_v_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*20/uz_8_20, PSD_v_seq_8_20*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*20/uz_5_20, PSD_v_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_v_20_1 = funcs.kaimal_v(f_seq_1[1:], uz_1_20, zSeq_1[0], uStar_1)
plt.loglog(f_seq_1[1:]*20/uz_1_20, kaimal_v_20_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_v/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sv_f_scaled_cmp_z0_h20.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Sv scaled cmp z0 - 180m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_180 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,8,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_180 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,8,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_180 = calc_uz_palm(8,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*180/uz_9_180, PSD_v_seq_9_180*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*180/uz_1_180, PSD_v_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*180/uz_8_180, PSD_v_seq_8_180*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*180/uz_5_180, PSD_v_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_v_180_1 = funcs.kaimal_v(f_seq_1[1:], uz_1_180, zSeq_1[8], uStar_1)
plt.loglog(f_seq_1[1:]*180/uz_1_180, kaimal_v_180_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_v/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sv_f_scaled_cmp_z0_h180.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()








""" Sw scaled cmp z0 - 100m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_100 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,4,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_100 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,4,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_100 = calc_uz_palm(4,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*100/uz_9_100, PSD_w_seq_9_100*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*100/uz_1_100, PSD_w_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*100/uz_8_100, PSD_w_seq_8_100*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*100/uz_5_100, PSD_w_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_w_100_1 = funcs.kaimal_w(f_seq_1[1:], uz_1_100, zSeq_1[4], uStar_1)
plt.loglog(f_seq_1[1:]*100/uz_1_100, kaimal_w_100_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_w/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sw_f_scaled_cmp_z0_h100.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Sw scaled cmp z0 - 20m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_20 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,0,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_20 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,0,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_20 = calc_uz_palm(0,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*20/uz_9_20, PSD_w_seq_9_20*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*20/uz_1_20, PSD_w_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*20/uz_8_20, PSD_w_seq_8_20*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*20/uz_5_20, PSD_w_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_w_20_1 = funcs.kaimal_w(f_seq_1[1:], uz_1_20, zSeq_1[0], uStar_1)
plt.loglog(f_seq_1[1:]*20/uz_1_20, kaimal_w_20_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_w/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sw_f_scaled_cmp_z0_h20.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()

""" Sw scaled cmp z0 - 180m """
## sowfa
# compute uz, uStar
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
uStar_8 = calc_uStar_sowfa(xSeq_8.size,ySeq_8.size,zSeq_8,uSeq_8,kappa=0.4,z0=0.0001)
uz_8_180 = calc_uz_sowfa(xSeq_8.size,ySeq_8.size,8,uSeq_8)
uStar_9 = calc_uStar_sowfa(xSeq_9.size,ySeq_9.size,zSeq_9,uSeq_9,kappa=0.4,z0=0.01)
uz_9_180 = calc_uz_sowfa(xSeq_9.size,ySeq_9.size,8,uSeq_9)

## palm
# compute uz, uStar
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.001)
uz_5_180 = calc_uz_palm(8,uSeq_5)

# plot
fig, ax = plt.subplots(figsize=(6,4))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]
plt.loglog(f_seq_9*180/uz_9_180, PSD_v_seq_9_180*f_seq_9/np.power(uStar_9,2), label='sowfa-0.01', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1*180/uz_1_180, PSD_v_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_8*180/uz_8_180, PSD_v_seq_8_180*f_seq_8/np.power(uStar_8,2), label='sowfa-0.0001', linewidth=1.0, linestyle='-', color=colors_sowfa[2])

plt.loglog(f_seq_5*180/uz_5_180, PSD_v_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.001', linewidth=1.0, linestyle='-', color=colors_palm[1])

# kaimal spectrum
kaimal_w_180_1 = funcs.kaimal_w(f_seq_1[1:], uz_1_180, zSeq_1[8], uStar_1)
plt.loglog(f_seq_1[1:]*180/uz_1_180, kaimal_w_180_1*f_seq_1[1:]/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
# plot -5/3 law
f_ = np.linspace(1e-1,1e1,100)
plt.loglog(f_, 1e1*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')

plt.xlabel(r"$\mathrm{fz/\overline{u}}$")
plt.ylabel(r"$\mathrm{fS_w/u^2_*}$")
xaxis_min = 1e-2
xaxis_max = 10 # f_seq.max()
yaxis_min = 1e-8
yaxis_max = 1e2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
# plt.legend(bbox_to_anchor=(0.05,0.2), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Sw_f_scaled_cmp_z0_h180.png'
plt.savefig('/scratch/projects/deepwind/photo/PSD.24.01.2021' + '/' + saveName, bbox_inches='tight')
plt.show()
