#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import scipy.signal
from funcs import *
import matplotlib.pyplot as plt


def velo_prb(dir, jobName, maskID, run_no_list, var):
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

def PSD(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

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
            tmp = window_weight(tmp)
            # FFT
            # omega_seq, tmp = PSD_omega(t_seq,tmp)
            f_seq, tmp = scipy.signal.csd(tmp, tmp, fs, nperseg=segNum, noverlap=None)
            PSD_list.append(tmp)
    PSD_seq = np.average(np.array(PSD_list), axis=0)

    return f_seq, PSD_seq





prjDir = '/scratch/palmdata/JOBS'


jobName  = 'deepwind_gs10'
dir_0 = prjDir + '/' + jobName
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0 = velo_prb(dir_0, jobName, 'M03', ['.022'], 'u')
f_seq, PSD_u_seq_0 = PSD((288000.0, 290400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, varSeq_0, 20480)

# plot PSD_u_f
fig, ax = plt.subplots(figsize=(8,5))
plt.loglog(f_seq, PSD_u_seq_0, label='', linewidth=1.0, color='b')
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)')
# plt.ylabel(varName + ' (' + varUnit + ')')
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
# saveName = varName_save + '_PSD_f_prb' + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
