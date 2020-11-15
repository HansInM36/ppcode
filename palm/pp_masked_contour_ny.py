#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

jobname  = 'pcr_NBL_U10_gs20'
cycle_no_list = ['.000'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

maskid = 'M02'

varname = 'u'
varunit = 'm/s'
tind = 0 # specify the index of time
yind = 0

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
for i in range(cycle_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['u'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var
zname = list(nc_file_list[0].variables[varname].dimensions)[1] # the height name string
zseq = np.array(nc_file_list[0].variables[zname][:], dtype=type(nc_file_list[0].variables[zname])) # array of height levels
yname = list(nc_file_list[0].variables[varname].dimensions)[2] # the height name string
yseq = np.array(nc_file_list[0].variables[yname][:], dtype=type(nc_file_list[0].variables[yname])) # array of height levels
xname = list(nc_file_list[0].variables[varname].dimensions)[3] # the height name string
xseq = np.array(nc_file_list[0].variables[xname][:], dtype=type(nc_file_list[0].variables[xname])) # array of height levels

# extract the part of the var and concatenate arraies of all cycle_no_list
varseq_list = []
for i in range(cycle_num):
    varseq_list.append(np.array(nc_file_list[i].variables[varname][tind,:,yind,:], dtype=type(nc_file_list[i].variables[varname])))
varseq = np.concatenate([varseq_list[i] for i in range(cycle_num)], axis=0)

### plot
z_start_ind = 1 # sometimes need to abandon the unphysical value at z=0
fig, axs = plt.subplots(figsize=(8,8))
cbreso = 100 # resolution of colorbar
CS = axs.contourf(xseq, zseq[z_start_ind:], varseq[z_start_ind:], cbreso, cmap='coolwarm')
cbar = plt.colorbar(CS, ax=axs, shrink=1.0)
cbar.ax.set_ylabel('u (m/s)')
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
saveName = varname + '_' + str(int(tseq[tind])) +'_contour_ny_' + str(int(yseq[yind])) + '.png'
plt.savefig(saveDir + saveName)
plt.show()
