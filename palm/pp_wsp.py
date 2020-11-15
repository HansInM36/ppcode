#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm


jobname  = 'pcr_NBL_U10_gs20'
cycle_no_list = ['.000'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

#time     = "7200"           # in s
#height   = "90"             # z-position of plot (nacelle at 90 m)
#xpos     = "1030"           # x-position of plot (turbine at 2000 m)
#ypos     = "2400"           # y-position of plot (turbine at 2000 m)

# some variables to choose for plotting: varname, axis, height(in plotting part)
varname = 'w_x'
varunit = 'm^3/s^2'
axis = 'x' # the direction of spectrum

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
varseq_list = []
for i in range(cycle_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_sp" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varseq_list.append(np.array(nc_file_list[i].variables[varname][:], dtype=type(nc_file_list[i].variables[varname])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['k_x'].dimensions)) #list dimensions of a specified variable

height = list(nc_file_list[0].variables[varname].dimensions)[1] # the height name string
hseq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
hlen = np.shape(hseq)[0]

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)
tlen = np.shape(tseq)[0]

axisseq = np.array(nc_file_list[0].variables['k_' + axis][:], dtype=type(nc_file_list[0].variables['k_' + axis])) # array of wave number
varseq = np.concatenate([varseq_list[i] for i in range(cycle_num)], axis=0)

# ### plot spectrum of various time at a certain height level
# hind = 2 # the index of height level
# fig, ax = plt.subplots(figsize=(8,6))
# colors = plt.cm.jet(np.linspace(0,1,tlen))
# delta = 3 # not necessary to plot data of all the time, so set time index interval delta
# for i in range(0,tlen,delta):
#     plt.loglog(axisseq, varseq[i][hind], label='t = ' + str(int(tseq[i])) + 's', linewidth=1.0, color=colors[i])
# plt.xlabel('k_x (m^-1)')
# plt.ylabel('S (m^3*s^-2)')
# plt.legend()
# plt.grid()
# # plt.show()
# saveDir = '/home/nx/Dropbox/UiB/report/12.10.2020/photo/' + jobname + '/'
# saveName = varname + '_' + 'h' + str(hseq[hind]) + '_sp.png'
# plt.savefig(saveDir + saveName)

### plot spectrum of various height levels at a certain time
tind = 0 # the index of time
fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.jet(np.linspace(0,1,hlen))
delta = 1 # not necessary to plot data of all the time, so set time index interval delta
for i in range(0,hlen,delta):
    plt.loglog(axisseq, varseq[tind][i], label='h = ' + str(int(hseq[i])) + 'm', linewidth=1.0, color=colors[i])
plt.xlabel('k_' + axis + ' (m^-1)')
plt.ylabel('S (' + varunit + ')')
plt.legend()
plt.grid()
plt.title(jobname)
fig.tight_layout() # adjust the layout
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = varname + '_' + 't' + str(int(tseq[tind])) + '_sp.png'
plt.savefig(saveDir + saveName)
plt.show()
