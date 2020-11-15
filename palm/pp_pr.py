#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

prjname  = 'pcr_NBL_U10'
suffix = '_gs20'
jobname = prjname + suffix

cycle_no_list = ['.000', '.001', '.002', '.004', '.005'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

varname = 'u'
varunit = 'm/s'

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
varseq_list = []
for i in range(cycle_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_pr" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varseq_list.append(np.array(nc_file_list[i].variables[varname][:], dtype=type(nc_file_list[i].variables[varname])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

height = list(nc_file_list[0].variables[varname].dimensions)[1] # the height name string
zseq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
zlen = zseq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)
varseq = np.concatenate([varseq_list[i] for i in range(cycle_num)], axis=0)

tlen = np.shape(tseq)[0]

### plot
tplot_start = 7200.0
tplot_end = 136800.0
tplot_delta = 7200.0

tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zlen)
    for zind in range(zlen):
        f = interp1d(tseq, varseq[:,zind])
        varplot[zind] = f(tplot)
    varplotList.append(varplot)


fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(varplotList[i], zseq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.axhline(y=102, ls='--', c='black')
plt.xlabel(varname + ' (' + varunit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 14
xaxis_d = 2
yaxis_min = 0
yaxis_max = 800.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveDir = '/scratch/sowfadata/pp/' + jobName + '/'
# saveName = varname + '_pr.png'
# plt.savefig(saveDir + saveName)
plt.show()



# tind_start = 2
# tind_end = 39
# tind_delta = 2 # not necessary to plot data of all the time, so set time index interval delta
# tindList = list(range(tind_start, tind_end + tind_delta, tind_delta))
# tindNum = int((tind_end - tind_start) / tind_delta + 1)
#
# fig, ax = plt.subplots(figsize=(6,6))
#
# colors = plt.cm.jet(np.linspace(0,1,tindNum))
# for i in range(tindNum):
#     plt.plot(varseq[tindList[i]], zseq, label='t = ' + str(int(tseq[tindList[i]])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=102, ls='--', c='black')
# plt.xlabel(varname + ' (' + varunit + ')')
# plt.ylabel('z (m)')
# xaxis_min = 0
# xaxis_max = 14
# xaxis_d = 2
# yaxis_min = 0
# yaxis_max = 800.0
# yaxis_d = 100.0
# plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
# plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.grid()
# plt.title('')
# fig.tight_layout() # adjust the layout
# saveDir = '/scratch/palmdata/pp/' + jobname + '/'
# saveName = varname + '_pr.png'
# plt.savefig(saveDir + saveName)
# plt.show()
#
# ### compute the horizontal velocity and direction at given height
# tind = tlen - 1
# f = interp1d(zseq.astype(float), varseq[tind].astype('float'), kind='cubic', fill_value="extrapolate")
# height = 102
# print(f(height))
