#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
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
varName = r'$Su_x$'
varUnit = r'$m^3/s^2$'
varName_save = 'Su_x'

tInd = -1

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

plotDataList = []
HList = []

for zInd in range(zNum):
    HList.append(zSeq[zInd])

    ESD_list = []

    for yInd in range(yNum):
        # detrend by deg_ order plynomial fit
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(xSeq, varSeq[tInd,zInd,yInd,:], deg=deg_))
        tmp = varSeq[tInd,zInd,yInd,:] - polyFunc(xSeq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = window_weight(tmp)
        kSeq, tmp = ESD_k(xSeq, tmp)
        ESD_list.append(tmp)

    # horizontal average
    ESD_seq = np.average(np.array(ESD_list), axis=0)
    plotDataList.append((kSeq, ESD_seq))

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,zNum))

for zInd in range(zNum):
    k_ = plotDataList[zInd][0]
    ESD_ = plotDataList[zInd][1]
    plt.loglog(k_, ESD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
# -5/3 law
plt.loglog(k_[1:], 100*np.power(k_[1:], -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('k (1/m)')
plt.ylabel(varName + ' (' + varUnit + ')')
xaxis_min = 1e-3
xaxis_max = kSeq.max()
yaxis_min = 1e-4
yaxis_max = 1e6
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)

plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_ESD_' + str(tSeq[tInd]) + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
