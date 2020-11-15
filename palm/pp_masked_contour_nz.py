#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

jobname  = 'pcr_NBL_U10_gs20'
rnList = ['.005'] # "" for initial run, ".001" for first cycle, etc.
run_num = len(rnList)

maskid = 'M01'

varname = 'u'
varunit = 'm/s'
tind = 0 # specify the index of time

zindList = [0,1,2,3,4,5,6,7] # specify the index of height level

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all rnList
file_rn = []
tseq_rn = []
for i in range(run_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_masked_" + maskid + rnList[i] + ".nc"
    file_rn.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_rn.append(np.array(file_rn[i].variables['time'][:], dtype=type(file_rn[i].variables['time'])))

# concatenate arraies of all rnList along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_rn[i] for i in range(run_num)], axis=0)

# print(list(file_rn[0].dimensions)) #list all dimensions
# print(list(file_rn[0].variables)) #list all the variables
# print(list(file_rn[0].variables['u'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var
zname = list(file_rn[0].variables[varname].dimensions)[1] # the height name string
zseq = np.array(file_rn[0].variables[zname][:], dtype=type(file_rn[0].variables[zname])) # array of height levels
yname = list(file_rn[0].variables[varname].dimensions)[2] # the height name string
yseq = np.array(file_rn[0].variables[yname][:], dtype=type(file_rn[0].variables[yname])) # array of height levels
xname = list(file_rn[0].variables[varname].dimensions)[3] # the height name string
xseq = np.array(file_rn[0].variables[xname][:], dtype=type(file_rn[0].variables[xname])) # array of height levels

zlen = len(zindList)
varseqList = [[] for zind in range(zlen)]

for zind in zindList:
    # extract the part of the var and concatenate arraies of all rnList
    varseq_rn = []
    for i in range(run_num):
        varseq_rn.append(np.array(file_rn[i].variables[varname][tind,zind,:,:], dtype=type(file_rn[i].variables[varname])))
    varseqList[zind] = np.concatenate([varseq_rn[i] for i in range(run_num)], axis=0)
    varseqList[zind] = varseqList[zind] - np.mean(varseqList[zind])

print(np.max(varseqList), np.min(varseqList))

fig, axs = plt.subplots(2,2)
fig.set_figwidth(8)
fig.set_figheight(8)

cbreso = 100 # resolution of colorbar
vmin = -2.0
vmax = 2.0
vdelta = 0.5

for zind in zindList:
    varseqList[zind][np.where(varseqList[zind] < vmin)] = vmin
    varseqList[zind][np.where(varseqList[zind] > vmax)] = vmax

levels = np.linspace(vmin, vmax, cbreso + 1)

# assign the return of first contourf plot for drawing color bar
clb = \
axs[0,0].contourf(xseq, yseq, varseqList[4], cbreso, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
axs[0,1].contourf(xseq, yseq, varseqList[5], cbreso, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
axs[1,0].contourf(xseq, yseq, varseqList[6], cbreso, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
axs[1,1].contourf(xseq, yseq, varseqList[7], cbreso, levels=levels, cmap='jet', vmin=vmin, vmax=vmax)
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,1].set_yticks([])
axs[1,0].set_xlabel('x (m)', fontsize=12)
axs[1,1].set_xlabel('x (m)', fontsize=12)
axs[0,0].set_ylabel('y (m)', fontsize=12)
axs[1,0].set_ylabel('y (m)', fontsize=12)
# height label
axs[0,0].text(0.6, 1.02, 'h = ' + str(int(zseq[4])) + 'm', transform=axs[0,0].transAxes, fontdict={'size':12})
axs[0,1].text(0.6, 1.02, 'h = ' + str(int(zseq[5])) + 'm', transform=axs[0,1].transAxes, fontdict={'size':12})
axs[1,0].text(0.6, 1.02, 'h = ' + str(int(zseq[6])) + 'm', transform=axs[1,0].transAxes, fontdict={'size':12})
axs[1,1].text(0.6, 1.02, 'h = ' + str(int(zseq[7])) + 'm', transform=axs[1,1].transAxes, fontdict={'size':12})

cbartickList = np.linspace(vmin, vmax, int((vmax-vmin)/vdelta)+1)
cbar = fig.colorbar(clb, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.set_label(varname + r' $(m \cdot s^{-1})$', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plotNo = 1 # give a number of this plot
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = varname + '_' + str(int(tseq[tind])) +'_contour_nz_' + str(plotNo) + '.png'
plt.savefig(saveDir + saveName, bbox_inches='tight')
plt.show()




### plot
fig, axs = plt.subplots(figsize=(8,8))
cbreso = 100 # resolution of colorbar
CS = axs.contourf(xseq, yseq, varseq, cbreso, cmap='coolwarm')
cbar = plt.colorbar(CS, ax=axs, shrink=1.0)
cbar.ax.set_ylabel('u (m/s)')
plt.text(0.8, 1.02, 'h = ' + str(int(zseq[zind])) + 'm', transform=axs.transAxes, fontdict={'size':12})
# xaxis_min = xseq
# xaxis_max = 2
# xaxis_d = 1
# yaxis_min = 0
# yaxis_max = 1000.0
# yaxis_d = 200
# plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.ylabel('y (m)')
plt.xlabel('x (m)')
# fig.tight_layout() # adjust the layout
plt.title(jobname)
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = varname + '_' + str(int(tseq[tind])) +'_contour_nz_' + str(int(zseq[zind])) + '.png'
plt.savefig(saveDir + saveName)
plt.show()
