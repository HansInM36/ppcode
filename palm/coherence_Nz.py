#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from funcs import *
import matplotlib.pyplot as plt

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_NBL_U10'
suffix = '_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M01'

cycle_no_list = ['.005'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

var = 'u'
varName = 'coherence'
varUnit = ''
varName_save = 'uu_coherence'


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
t_end = 435600.0
t_delta = 2.0
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


zInd = 0
for zInd in range(zNum):
    # ptCoorList = [(49,50),(50,50),(51,50),(50,49),(50,51)]
    ptCoorList = [(45,50),(50,50),(55,50),(50,45),(50,55)]
    ptNum = len(ptCoorList)

    v_seq_list = []

    for pt in range(ptNum):
        vSeq = varSeq[:,zInd,ptCoorList[pt][1],ptCoorList[pt][0]]
        f = interp1d(tSeq, vSeq, fill_value='extrapolate')
        v_seq = f(t_seq)
        v_seq = v_seq
        v_seq_list.append(v_seq)

    freq, coh, phase = coherence(v_seq_list[1], v_seq_list[1], 0.5)
    freq0, coh0, phase = coherence(v_seq_list[0], v_seq_list[1], 0.5)
    freq1, coh1, phase = coherence(v_seq_list[0], v_seq_list[2], 0.5)
    freq2, coh2, phase = coherence(v_seq_list[3], v_seq_list[1], 0.5)
    freq3, coh3, phase = coherence(v_seq_list[3], v_seq_list[4], 0.5)


    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.jet(np.linspace(0,1,4))

    # for zInd in range(zNum):
    #     f_ = plotDataList[zInd][0] / (2*np.pi) # convert from omega to frequency
    #     ESD_ = plotDataList[zInd][1] * 2*np.pi
    #     plt.loglog(f_, ESD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
    # -5/3 law
    plt.plot(freq0, coh0, label='coh01', linewidth=1.0, color=colors[0])
    plt.plot(freq1, coh1, label='coh02', linewidth=1.0, color=colors[1])
    plt.plot(freq2, coh2, label='coh31', linewidth=1.0, color=colors[2])
    plt.plot(freq3, coh3, label='coh34', linewidth=1.0, color=colors[3])
    plt.xlabel('f (1/s)')
    plt.ylabel(varName)
    xaxis_min = 0
    xaxis_max = 0.25
    yaxis_min = 0
    yaxis_max = 1
    plt.ylim(yaxis_min, yaxis_max)
    plt.xlim(xaxis_min, xaxis_max)
    plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('')
    fig.tight_layout() # adjust the layout
    saveName = varName_save + '_' + str(zSeq[zInd]) + '.png'
    plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
