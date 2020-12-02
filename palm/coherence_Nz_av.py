#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import funcs
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
dx = xSeq[1] - xSeq[0]
xNum = xSeq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

t_start = 432000.0
t_end = 435600.0
t_delta = 2.0
fs = 1/t_delta
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)

dIndList = [5,10,15,20]
dIndNum = len(dIndList)


def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)


### calculate horizontally averaged uu_x coherence
plotDataList = []

for zInd in range(zNum):

    cohList = []
    for dInd in dIndList:

        xInd_start = 25
        xInd_end = xNum - 26
        yInd_start = 25
        yInd_end = yNum - 26

        coh = []
        yInd = yInd_start
        while yInd <= yInd_end:
            coh_temp = []
            xInd0 = xInd_start
            xInd1 = xInd0 + dInd
            while xInd1 < xNum:
                u0 = varSeq[:,zInd,yInd,xInd0]
                u1 = varSeq[:,zInd,yInd,xInd1]
                # time interpolation
                f0 = interp1d(tSeq, u0, kind='cubic', fill_value='extrapolate')
                f1 = interp1d(tSeq, u1, kind='cubic', fill_value='extrapolate')
                u0 = f0(t_seq)
                u1 = f1(t_seq)
                # calculate coherence and phase
                freq, coh_, phase_ = funcs.coherence(u0, u1, fs)

                coh_temp.append(coh_)

                xInd0 += 1
                xInd1 += 1

            coh_temp = sum(coh_temp)/len(coh_temp)

            coh.append(coh_temp)

            yInd += 1

        coh = sum(coh)/len(coh)
        cohList.append(coh)

    plotDataList.append(cohList)


for i in range(zNum):

    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.jet(np.linspace(0,1,dIndNum))

    for j in range(dIndNum):
        coh = plotDataList[i][j]
        ax.plot(freq, coh, label='dx = ' + str(dx*dIndList[j]) + 'm', linestyle='', marker='o', markersize=3, color=colors[j])
        ind_in, ind_out = 0, 77
        popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 9999]))
        ax.plot(freq[ind_in:ind_out], fitting_func(freq[ind_in:ind_out], *popt), linestyle=':', color=colors[j],
             label='a=%5.3f, alpha=%5.3f' % tuple(popt))

    plt.xlabel('f (1/s)')
    plt.ylabel('Coherence')
    # xaxis_min = 5
    # xaxis_max = 10
    # xaxis_d = 0.5
    yaxis_min = 0
    yaxis_max = 1.0
    yaxis_d = 0.1
    plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
    # plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
    # plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.text(0.8, 1.02, 'h = ' + str(int(zSeq[i])) + 'm', transform=ax.transAxes, fontsize=12)
    plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('')
    fig.tight_layout() # adjust the layout
    saveName = varName_save + '_' + str(int(zSeq[i])) + '_pr.png'
    # plt.savefig(ppDir + '/' + saveName)
    plt.show()
