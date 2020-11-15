#!/usr/bin/python3.8

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_NBL_U10'
suffix = '_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M02'

cycle_no_list = ['.005'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

varName = 'u'
varUnit = 'm/s'

tInd = -1

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
varSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varSeq_list.append(np.array(nc_file_list[i].variables[varName][:], dtype=type(nc_file_list[i].variables[varName])))

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
yNum = ySeq.size
xName = list(nc_file_list[0].variables[varName].dimensions)[3] # the height name string
xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
xSeq = xSeq.astype(float)

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

plotDataList = []
YList = []

for yInd in range(yNum):
    YList.append(ySeq[yInd])
    plotDataList.append((xSeq, zSeq, varSeq[tInd,:,yInd,:]))

### group plot
rNum, cNum = (1,1)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=True)

fig.set_figwidth(8)
fig.set_figheight(6)

cbreso = 100 # resolution of colorbar
vMin, vMax, vDelta = (-1.4, 1.2, 0.2)
levels = np.linspace(vMin, vMax, cbreso + 1)

for i in range(yNum):
    rNo = int(np.floor(i/cNum))
    cNo = int(i - rNo*cNum)
    x_ = plotDataList[i][0]
    z_ = plotDataList[i][1]
    v_ = plotDataList[i][2]
    for zInd in range(z_.size):
        v_[zInd] = v_[zInd] - v_[zInd].mean()
    # clb is for creating a common colorbar
    if rNum == 1 and cNum == 1:
        clb = axs.contourf(x_, z_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
        axs.set_ylim(0,600)
        axs.set_xlabel('x (m)', fontsize=12)
        axs.set_ylabel('z (m)', fontsize=12)
        axs.text(0.8, 1.02, 'y = ' + str(int(YList[i])) + 'm', transform=axs.transAxes, fontsize=12)
    else:
        clb = axs[rNo,cNo].contourf(x_, z_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
        if rNo == rNum - 1:
            axs[rNo,cNo].set_xlabel('x (m)', fontsize=12)
        else:
            axs[rNo,cNo].set_xticks([])
        if cNo == 0:
            axs[rNo,cNo].set_ylabel('z (m)', fontsize=12)
        else:
            axs[rNo,cNo].set_yticks([])
        axs[rNo,cNo].text(0.6, 1.02, 'y = ' + str(int(YList[i])) + 'm', transform=axs[rNo,cNo].transAxes, fontsize=12)
cbartickList = np.linspace(vMin, vMax, int(np.round((vMax-vMin)/vDelta))+1)
cbar = fig.colorbar(clb, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.set_label(varName + ' (m/s)', fontsize=12)
cbar.ax.tick_params(labelsize=12)
fig.suptitle('t = ' + str(np.round(tSeq[tInd],2)) + 's')
saveName = varName + '_contour_' + str(tSeq[tInd]) + '_' + str(int(ySeq[0])) + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()


# ### single plot
# sliceInd = 2
# fig, axs = plt.subplots(figsize=(8,8), constrained_layout=True)
# cbreso = 100 # resolution of colorbar
# x_ = plotDataList[sliceInd][0]
# y_ = plotDataList[sliceInd][1]
# v_ = plotDataList[sliceInd][2]
# CS = axs.contourf(x_, y_, v_ - v_.mean(), cbreso, cmap='jet')
# cbar = plt.colorbar(CS, ax=axs, shrink=1.0)
# cbar.ax.set_ylabel(varNameDict[varD] + ' (m/s)', fontsize=12)
# plt.ylabel('y (m)')
# plt.xlabel('x (m)')
# plt.title('t = ' + str(tSeq[tInd]) + 's')
# saveName = varNameDict[varD] + '_contour_' + str(tSeq[tInd]) + '_' + sliceList[sliceInd] + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
# plt.show()
