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

cycle_no_list = ['.000','.001','.002'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

rsv = 'w*u*'
sgs = 'w"u"'
varName = "u-component vertical momentum flux"
varName_save = 'cfl_uw_flux'
varUnit = r'$\mathrm{m^2/s^2}$'

hubH = 90

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
rsvSeq_list = []
sgsSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_pr" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    rsvSeq_list.append(np.array(nc_file_list[i].variables[rsv][:], dtype=type(nc_file_list[i].variables[rsv])))
    sgsSeq_list.append(np.array(nc_file_list[i].variables[sgs][:], dtype=type(nc_file_list[i].variables[sgs])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

height = list(nc_file_list[0].variables[rsv].dimensions)[1] # the height name string
zSeq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
zSeq = zSeq.astype(float)
zNum = zSeq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
tNum = tSeq.size

rsvSeq = np.concatenate([rsvSeq_list[i] for i in range(cycle_num)], axis=0)
rsvSeq = rsvSeq.astype(float)
sgsSeq = np.concatenate([sgsSeq_list[i] for i in range(cycle_num)], axis=0)
sgsSeq = sgsSeq.astype(float)



### plot
# ave_itv = 3600.0 # by default, the averaging interval is 3600s
tplot = 432000.0

rsvplot = np.zeros(zNum)
sgsplot = np.zeros(zNum)
for zInd in range(zNum):
    f = interp1d(tSeq[1:], rsvSeq[1:,zInd], kind='linear', fill_value='extrapolate')
    rsvplot[zInd] = f(tplot)

    f = interp1d(tSeq[1:], sgsSeq[1:,zInd], kind='linear', fill_value='extrapolate')
    sgsplot[zInd] = f(tplot)

cflInd = 9
fig, ax = plt.subplots(figsize=(6,6))
plt.plot(rsvplot[:cflInd], zSeq[:cflInd], label='resolved', linestyle='--', linewidth=1.0, color='r')
plt.plot(sgsplot[:cflInd], zSeq[:cflInd], label='SGS', linestyle=':', linewidth=1.0, color='b')
plt.plot(rsvplot[:cflInd]+sgsplot[:cflInd], zSeq[:cflInd], label='total', linestyle='-', linewidth=1.0, color='k')
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(varName + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = -0.1
xaxis_max = 0.02
xaxis_d = 0.02
yaxis_min = 0
yaxis_max = 180.0
yaxis_d = 20.0
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
