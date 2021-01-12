#!/usr/bin/python3.8

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind'
suffix = '_gs10'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M01'

cycle_no_list = ['.021','.022'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

varName = 'u'
varUnit = 'm/s'

tInd = -1

xInd_start, xInd_end = 40, 91
yInd_start, yInd_end = 40, 91

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
varSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varSeq_list.append(np.array(nc_file_list[i].variables[varName][:,:,yInd_start:yInd_end,xInd_start:xInd_end], dtype=type(nc_file_list[i].variables[varName])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var
zName = list(nc_file_list[0].variables[varName].dimensions)[1] # the height name string
zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
zNum = zSeq.size
zSeq = zSeq.astype(float)
yName = list(nc_file_list[0].variables[varName].dimensions)[2] # the height name string
ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
ySeq = ySeq.astype(float)
xName = list(nc_file_list[0].variables[varName].dimensions)[3] # the height name string
xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
xSeq = xSeq.astype(float)

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
tNum = tSeq.size
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

plotDataList = []
HList = []

for zInd in range(zNum):
    HList.append(zSeq[zInd])
    plotDataList.append((xSeq[xInd_start:xInd_end], ySeq[yInd_start:yInd_end], varSeq[tInd,zInd]))

### group plot
rNum, cNum = (4,2)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=True)

fig.set_figwidth(8)
fig.set_figheight(12)

cbreso = 100 # resolution of colorbar
vMin, vMax, vDelta = (-2.4, 2.4, 0.4)
levels = np.linspace(vMin, vMax, cbreso + 1)

for i in range(zNum):
    rNo = int(np.floor(i/cNum))
    cNo = int(i - rNo*cNum)
    x_ = plotDataList[i][0]
    y_ = plotDataList[i][1]
    v_ = plotDataList[i][2]
    # clb is for creating a common colorbar
    clb = axs[rNo,cNo].contourf(x_, y_, v_ - v_.mean(), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    if rNo == rNum - 1:
        axs[rNo,cNo].set_xlabel('x (m)', fontsize=12)
    else:
        axs[rNo,cNo].set_xticks([])
    if cNo == 0:
        axs[rNo,cNo].set_ylabel('y (m)', fontsize=12)
    else:
        axs[rNo,cNo].set_yticks([])
    axs[rNo,cNo].text(0.6, 1.02, 'h = ' + str(int(HList[i])) + 'm', transform=axs[rNo,cNo].transAxes, fontsize=12)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = fig.colorbar(clb, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.set_label(varName + ' (m/s)', fontsize=12)
cbar.ax.tick_params(labelsize=12)
fig.suptitle('t = ' + str(np.round(tSeq[tInd],2)) + 's')
saveName = varName + '_contour_' + str(tSeq[tInd]) + '_' + str(int(zSeq[0])) + '_etc' + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()





zInd = 2

plotDataList = []
HList = []

for tInd in range(tNum):
    HList.append(zSeq[zInd])
    plotDataList.append((xSeq[xInd_start:xInd_end], ySeq[yInd_start:yInd_end], varSeq[tInd,zInd]))


### single plot
vMin, vMax, vDelta = (-1.8, 1.8, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(0,600,10):
    fig, axs = plt.subplots(figsize=(8,8), constrained_layout=False)
    x_ = plotDataList[tInd][0]
    y_ = plotDataList[tInd][1]
    v_ = plotDataList[tInd][2]
    CS = axs.contourf(x_, y_, v_ - v_.mean(), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd],2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$u'$" + ' (m/s)', fontsize=12)
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.title('h = ' + str(int(HList[zInd])) + 'm')
    # saveName = varNameDict[varD] + '_contour_' + str(tSeq[tInd]) + '_' + sliceList[zInd] + '.png'
    saveName = "%.4d" % tInd + '.png'
    # plt.savefig(ppDir + '/animation/' + saveName, bbox_inches='tight')
    plt.savefig(ppDir + '/animation/' + saveName)
    # plt.show()
    plt.close()
