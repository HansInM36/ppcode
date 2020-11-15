#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

jobname  = 'pcr_NBL_U10_gs20'
rnList = ['.000','.001','.002','.004'] # ".000" for first run, ".001" for second run, etc.
run_num = len(rnList)

varnameList = ['time','umax','vmax','E','E*']
varunitList = ['s','m/s','m/s','m^2/s^2','m^2/s^2']
varnum = len(varnameList)

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_cn
file_rn = []

varseqList_cn = [[] for var in range(varnum)]

for i in range(run_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_ts" + rnList[i] + ".nc"
    file_rn.append(Dataset(input_file, "r", format="NETCDF4"))

    for var in range(varnum):
        varseqList_cn[var].append(np.array(file_rn[i].variables[varnameList[var]][:], dtype=type(file_rn[i].variables[varnameList[var]])))

varseqList = [[] for var in range(varnum)]
for var in range(varnum):
    varseqList[var] = np.concatenate([varseqList_cn[var][i] for i in range(run_num)], axis=0)

# print(list(file_rn[0].dimensions)) #list all dimensions
# print(list(file_rn[0].variables)) #list all the variables
# print(list(file_rn[0].variables['u2'].dimensions)) #list dimensions of a specified variable

tseq = varseqList[0]

fig, axs = plt.subplots(varnum-1,1)
fig.set_figwidth(8)
fig.set_figheight(9.6)

# colors = plt.cm.jet(np.linspace(0,1,varnum-1))
colors = ['g','b','r','k']
linestyles = ['-', '--', ':', '-.']

for i in range(0,axs.size):
    axs[i].plot(tseq, varseqList[i+1], label='', linewidth=1.0, color=colors[i])
    # axs[i].xlabel()
    axs[i].set_ylabel(varnameList[i+1] + ' (' + varunitList[i+1] + ')', fontsize=12)
    axs[i].grid()
    if i == axs.size - 1:
        axs[i].set_xlabel('time (s)', fontsize=12)
fig.tight_layout() # adjust the layout


saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = 'ts.png'
plt.savefig(saveDir + saveName)
plt.show()
