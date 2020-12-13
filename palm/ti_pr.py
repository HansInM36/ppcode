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

varName = r'$\mathrm{TI_u}$'
varUnit = '%'
varName_save = 'TI_u'

hubH = 90.0

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
uSeq_list = []
uuSeq_list = []
vSeq_list = []
vvSeq_list = []
wSeq_list = []
wwSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_pr" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    uSeq_list.append(np.array(nc_file_list[i].variables['u'][:], dtype=type(nc_file_list[i].variables['u'])))
    uuSeq_list.append(np.array(nc_file_list[i].variables['u*2'][:], dtype=type(nc_file_list[i].variables['u*2'])))
    vSeq_list.append(np.array(nc_file_list[i].variables['v'][:], dtype=type(nc_file_list[i].variables['v'])))
    vvSeq_list.append(np.array(nc_file_list[i].variables['v*2'][:], dtype=type(nc_file_list[i].variables['v*2'])))
    wSeq_list.append(np.array(nc_file_list[i].variables['w'][:], dtype=type(nc_file_list[i].variables['w'])))
    wwSeq_list.append(np.array(nc_file_list[i].variables['w*2'][:], dtype=type(nc_file_list[i].variables['w*2'])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

height = list(nc_file_list[0].variables['u'].dimensions)[1] # the height name string
zSeq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
zSeq = zSeq.astype(float)
zNum = zSeq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
uSeq = np.concatenate([uSeq_list[i] for i in range(cycle_num)], axis=0)
uSeq = uSeq.astype(float)
uuSeq = np.concatenate([uuSeq_list[i] for i in range(cycle_num)], axis=0)
uuSeq = uuSeq.astype(float)
vSeq = np.concatenate([vSeq_list[i] for i in range(cycle_num)], axis=0)
vSeq = vSeq.astype(float)
vvSeq = np.concatenate([vvSeq_list[i] for i in range(cycle_num)], axis=0)
vvSeq = vvSeq.astype(float)
wSeq = np.concatenate([wSeq_list[i] for i in range(cycle_num)], axis=0)
wSeq = wSeq.astype(float)
wwSeq = np.concatenate([wwSeq_list[i] for i in range(cycle_num)], axis=0)
wwSeq = wwSeq.astype(float)

### plot
varSeq = 100 * np.power(uuSeq[2:,1:], 0.5) / uSeq[2:,1:] # somehow negative var2 values appear in the first two time steps; TI(z=0) should be specified to 0

tplot_start = 3600.0*6
tplot_end = 3600.0*6*20
tplot_delta = 3600.0*6

tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum-1)
    for zInd in range(zNum-1):
        f = interp1d(tSeq[2:], varSeq[:,zInd], kind='linear', fill_value="extrapolate")
        varplot[zInd] = f(tplot)
    varplotList.append(varplot)


fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    zero = np.zeros(1)
    v_ = np.concatenate((zero, varplotList[i]))
    plt.plot(v_, zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
    # plt.plot(varplotList[i], zSeq[1:], label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(varName + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
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



### TI in 3 dimensions at certain timestep
TIuSeq = 100 * np.power(uuSeq[2:,1:], 0.5) / uSeq[2:,1:] # somehow negative var2 values appear in the first two time steps; TI(z=0) should be specified to 0
TIvSeq = 100 * np.power(vvSeq[2:,1:], 0.5) / uSeq[2:,1:]
TIwSeq = 100 * np.power(wwSeq[2:,1:], 0.5) / uSeq[2:,1:]

tplot = 432000.0

TIuplot = np.zeros(zNum-1)
TIvplot = np.zeros(zNum-1)
TIwplot = np.zeros(zNum-1)
for zInd in range(zNum-1):
    f = interp1d(tSeq[2:], TIuSeq[:,zInd], kind='linear', fill_value='extrapolate')
    # tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIuplot[zInd] = f(tplot)

    f = interp1d(tSeq[2:], TIvSeq[:,zInd], kind='linear', fill_value='extrapolate')
    # tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIvplot[zInd] = f(tplot)

    f = interp1d(tSeq[2:], TIwSeq[:,zInd], kind='linear', fill_value='extrapolate')
    # tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIwplot[zInd] = f(tplot)



fig, ax = plt.subplots(figsize=(6,6))
zero = np.zeros(1)
# TIuplot_ = np.concatenate((zero, TIuplot))
# TIvplot_ = np.concatenate((zero, TIvplot))
# TIwplot_ = np.concatenate((zero, TIwplot))
plt.plot(TIuplot, zSeq[1:], label='TIu', linewidth=1.0, color='r')
plt.plot(TIvplot, zSeq[1:], label='TIv', linewidth=1.0, color='b')
plt.plot(TIwplot, zSeq[1:], label='TIw', linewidth=1.0, color='g')
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel('TI' + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.8,0.9), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'TI_uvw' + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
