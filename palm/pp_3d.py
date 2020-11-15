#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

prjDir = "/scratch/palmdata"
jobname  = 'pcr_SBL_U10'
cycle_no_list = ['.002'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

#time     = "7200"           # in s
#height   = "90"             # z-position of plot (nacelle at 90 m)
#xpos     = "1030"           # x-position of plot (turbine at 2000 m)
#ypos     = "2400"           # y-position of plot (turbine at 2000 m)

varname = 'u'
varunit = 'm/s'

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
varseq_list = []
for i in range(cycle_num):
    input_file = prjDir + "/JOBS/" + jobname + "/OUTPUT/" + jobname + "_3d" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varseq_list.append(np.array(nc_file_list[i].variables[varname][:], dtype=type(nc_file_list[i].variables[varname])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)
varseq = np.concatenate([varseq_list[i] for i in range(cycle_num)], axis=0)

fig, ax = plt.subplots(figsize=(8,2.4))
plt.plot(tseq, varseq, label='', linewidth=1.0, color='k')
plt.ylabel(varname + ' (' + varunit + ')')
plt.xlabel('time (s)')
xaxis_min = 0
xaxis_max = 108000
xaxis_d = 10800
yaxis_min = 0
yaxis_max = 0.08
yaxis_d = 0.02
plt.xlim(xaxis_min, xaxis_max)
plt.ylim(yaxis_min, yaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.grid()
plt.title(jobname)
fig.tight_layout() # adjust the layout
saveDir = '/home/nx/Dropbox/UiB/report/12.10.2020/photo/' + jobname + '/'
saveName = varname + '_ts.png'
plt.savefig(saveDir + saveName)
plt.show()
