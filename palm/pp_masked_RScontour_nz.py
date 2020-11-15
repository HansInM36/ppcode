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

var1 = 'v'
var2 = 'w'
RS = 'RS23'
RSname = r"$\overline{v' w'}$"

zindList = [0,1,2,3,4,5,6,7]

# read the output data of all rnList
file_rn = []
tseq_rn = []
for i in range(run_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_masked_" + maskid + rnList[i] + ".nc"
    file_rn.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_rn.append(np.array(file_rn[i].variables['time'][:], dtype=type(file_rn[i].variables['time'])))

# concatenate arraies of all rnList along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_rn[i] for i in range(run_num)], axis=0)
tlen = tseq.size

# print(list(file_rn[0].dimensions)) #list all dimensions
# print(list(file_rn[0].variables)) #list all the variables
# print(list(file_rn[0].variables['u'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var1
zname = list(file_rn[0].variables[var1].dimensions)[1] # the height name string
zseq = np.array(file_rn[0].variables[zname][:], dtype=type(file_rn[0].variables[zname])) # array of height levels
yname = list(file_rn[0].variables[var1].dimensions)[2] # the height name string
yseq = np.array(file_rn[0].variables[yname][:], dtype=type(file_rn[0].variables[yname])) # array of height levels
xname = list(file_rn[0].variables[var1].dimensions)[3] # the height name string
xseq = np.array(file_rn[0].variables[xname][:], dtype=type(file_rn[0].variables[xname])) # array of height levels

zlen = len(zindList)

rsList = [[] for zind in range(zlen)]

for zind in zindList:
# zind = 0 # specify the index of height level

    print("+++ Preparing plots for h = ", zseq[zind], " ...")

    # create 0 array for calculating the temporal average value matrix of var1 and var2
    var1_av = np.zeros((xseq.size,yseq.size))
    var2_av = np.zeros((xseq.size,yseq.size))
    # calculate average
    for tind in range(0,tlen):
        var1seq_list = []
        var2seq_list = []
        for i in range(run_num):
            var1seq_list.append(np.array(file_rn[i].variables[var1][tind,zind,:,:], dtype=type(file_rn[i].variables[var1])))
            var2seq_list.append(np.array(file_rn[i].variables[var2][tind,zind,:,:], dtype=type(file_rn[i].variables[var2])))
        var1seq = np.concatenate([var1seq_list[i] for i in range(run_num)], axis=0)
        var2seq = np.concatenate([var2seq_list[i] for i in range(run_num)], axis=0)
        var1_av = var1_av + var1seq / tlen
        var2_av = var2_av + var2seq / tlen

    # create 0 array for calculating the Reynolds stress matrix of var1 and var2
    rs = np.zeros((xseq.size,yseq.size))
    varvar1 = np.zeros((xseq.size,yseq.size))
    varvar2 = np.zeros((xseq.size,yseq.size))
    for tind in range(0,tlen):
        var1seq_list = []
        var2seq_list = []
        for i in range(run_num):
            var1seq_list.append(np.array(file_rn[i].variables[var1][tind,zind,:,:], dtype=type(file_rn[i].variables[var1])))
            var2seq_list.append(np.array(file_rn[i].variables[var2][tind,zind,:,:], dtype=type(file_rn[i].variables[var2])))
        var1seq = np.concatenate([var1seq_list[i] for i in range(run_num)], axis=0)
        var2seq = np.concatenate([var2seq_list[i] for i in range(run_num)], axis=0)
        rs = rs + (var1seq - var1_av) * (var2seq - var2_av) / tlen
        varvar1 = varvar1 + np.power((var1seq - var1_av), 2) / tlen
        varvar2 = varvar2 + np.power((var2seq - var2_av), 2) / tlen
    # rs = rs / (np.power(varvar1,0.5) * np.power(varvar2,0.5)) # normalization

    rsList[zind] = rs

print(np.max(rsList), np.min(rsList))

fig, axs = plt.subplots(2,2)
fig.set_figwidth(8)
fig.set_figheight(8)

cbreso = 100 # resolution of colorbar
vmin = -0.04
vmax = 0.04
vdelta = 0.02
levels = np.linspace(vmin, vmax, cbreso + 1)

# assign the return of first contourf plot for drawing color bar
clb = \
axs[0,0].contourf(xseq, yseq, rsList[4], cbreso, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[0,1].contourf(xseq, yseq, rsList[5], cbreso, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[1,0].contourf(xseq, yseq, rsList[6], cbreso, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[1,1].contourf(xseq, yseq, rsList[7], cbreso, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
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
cbar.set_label(RSname + r' $(m^2 \cdot s^{-2})$', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plotNo = 1 # give a number of this plot
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = RS +'_contour_nz_' + str(plotNo) + '.png'
# plt.savefig(saveDir + saveName, bbox_inches='tight')
plt.show()
