#!/usr/bin/python3.8

import os
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind'
suffix = '_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

cycle_no_list = ['.000', '.001', '.002'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

varName = 'wind direction'
var1Name = 'u'
var2Name = 'v'
varUnit = r'$\degree$'
varName_save = 'wind_dir'

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
var1Seq_list = []
var2Seq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_pr" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    var1Seq_list.append(np.array(nc_file_list[i].variables[var1Name][:], dtype=type(nc_file_list[i].variables[var1Name])))
    var2Seq_list.append(np.array(nc_file_list[i].variables[var2Name][:], dtype=type(nc_file_list[i].variables[var2Name])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

height = list(nc_file_list[0].variables[var1Name].dimensions)[1] # the height name string
zSeq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
zSeq = zSeq.astype(float)
zNum = zSeq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
tNum = np.shape(tSeq)[0]
var1Seq = np.concatenate([var1Seq_list[i] for i in range(cycle_num)], axis=0)
var1Seq = var1Seq.astype(float)
var2Seq = np.concatenate([var2Seq_list[i] for i in range(cycle_num)], axis=0)
var2Seq = var2Seq.astype(float)

varSeq = 270 - np.arctan(var2Seq[:,1:] / var1Seq[:,1:]) * 180/np.pi

zSeq = zSeq[1:]
zNum = zSeq.size

### plot
tplot_start = 3600.0*8
tplot_end = 432000.0
tplot_delta = 3600.0*8

tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, varSeq[:,zInd], kind='cubic')
        varplot[zInd] = f(tplot)
    varplotList.append(varplot)

# plot
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(varName + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = 260
xaxis_max = 290
xaxis_d = 5
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
