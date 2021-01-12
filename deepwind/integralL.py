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
def getInfo_palm(dir, jobName, maskID, run_no, var):
    """ get information of x,y,z,t to decide how much data we should extract """
    input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no + ".nc"
    input_file = Dataset(input_file, "r", format="NETCDF4")
    zName = list(input_file.variables[var].dimensions)[1]
    zSeq = np.array(input_file.variables[zName][:], dtype=type(input_file.variables[zName]))
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(input_file.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_file.variables[yName][:], dtype=type(input_file.variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(input_file.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_file.variables[xName][:], dtype=type(input_file.variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size
    tSeq = np.array(input_file.variables['time'][:], dtype=type(input_file.variables['time']))
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    print('xMin = ', xSeq[0], ', ', 'xMax = ', xSeq[-1], ', ', 'xNum = ', xNum)
    print('yMin = ', ySeq[0], ', ', 'yMax = ', ySeq[-1], ', ', 'yNum = ', yNum)
    print('zMin = ', zSeq[0], ', ', 'zMax = ', zSeq[-1], ', ', 'zNum = ', zNum)
    print('tMin = ', tSeq[0], ', ', 'tMax = ', tSeq[-1], ', ', 'tNum = ', tNum)
    return tSeq, xSeq, ySeq, zSeq
def getData_palm(dir, jobName, maskID, run_no_list, var, tInd, xInd, yInd, zInd):
    """ extract velocity data of specified probe groups """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []

    tInd_start = 0
    list_num = 0
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1

        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][zInd[0]:zInd[1]], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][yInd[0]:yInd[1]], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][xInd[0]:xInd[1]], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(list_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq

jobName  = 'deepwind_gs10'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName


tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.022', 'u')
h = [zSeq[i] for i in range(zSeq.size)]
L = []
for zInd in range(zSeq.size):
    print(zInd)
    tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.022','.023'], 'u', (0,1200), (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    tNum, yNum, d = tSeq.size, ySeq.size, 10

    R = 0
    for tInd in range(tNum):
        for yInd in range(yNum):
            uSeq = varSeq[tInd,0,yInd,:]
            xi, R_ = funcs.autocorr_FFT(uSeq, d)
            R += R_
    R /= tNum*yNum
    xi, R = xi[0:xi.size//2], R[0:R.size//2]

    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.plot(xi,R)
    # plt.grid()
    # plt.show()

    Ind0 = 0
    for i in range(R.size):
        if R[i] < 0:
            Ind0 = i-1
            break
    L_ = (np.sum(R[0:Ind0]) - 0.5*(R[0]+R[Ind0])) * d
    L.append(L_)






























### compute integral length scale using probes
# zInd, yInd, d = 0, 0, 20
# varSeq = uSeq_0
#
# pInd_start = xSeq_0.size*ySeq_0.size*zInd + xSeq_0.size*yInd
# pInd_end = xSeq_0.size*ySeq_0.size*zInd + xSeq_0.size*(yInd+1)
# N = pInd_end - pInd_start
# tNum = varSeq.shape[1]
#
# s = d * np.arange(0,N)
# R = np.zeros(N)
#
# u0 = varSeq[pInd_start]
# u0 -= u0.mean()
# for p in range(N):
#     u1 = varSeq[pInd_start+p]
#     u1 -= u1.mean()
#     R[p] = np.mean(u0 * u1) / (np.std(u0)*np.std(u1))
#
# L = np.sum(R)*d - 0.5*(R[0] + R[-1])
#
# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(s,R)
# plt.grid()
# plt.show()
