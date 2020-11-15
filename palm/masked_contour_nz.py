#!/usr/bin/python3.8

import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

prjname  = 'cnp'
suffix = '_half'
jobname = prjname + suffix

maskid = 'M02'

cycle_no_list = ['.000'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

varname = 'u'
varunit = 'm/s'

tind = 1
zind = 2

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
varseq_list = []
for i in range(cycle_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varseq_list.append(np.array(nc_file_list[i].variables[varname][:], dtype=type(nc_file_list[i].variables[varname])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var
zname = list(nc_file_list[0].variables[varname].dimensions)[1] # the height name string
zseq = np.array(nc_file_list[0].variables[zname][:], dtype=type(nc_file_list[0].variables[zname])) # array of height levels
yname = list(nc_file_list[0].variables[varname].dimensions)[2] # the height name string
yseq = np.array(nc_file_list[0].variables[yname][:], dtype=type(nc_file_list[0].variables[yname])) # array of height levels
xname = list(nc_file_list[0].variables[varname].dimensions)[3] # the height name string
xseq = np.array(nc_file_list[0].variables[xname][:], dtype=type(nc_file_list[0].variables[xname])) # array of height levels


# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)
varseq = np.concatenate([varseq_list[i] for i in range(cycle_num)], axis=0)

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(8)

cbreso = 100 # resolution of colorbar
vmin = -2.0
vmax = 2.0
vdelta = 1.0
levels = np.linspace(vmin, vmax, cbreso + 1)

clb = axs.contourf(xseq, yseq, varseq[tind,zind,:,:], cbreso, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)

cbartickList = np.linspace(vmin, vmax, int((vmax-vmin)/vdelta)+1)
cbar = fig.colorbar(clb, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.set_label(varname + r' $(m \cdot s^{-1})$', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plotNo = 0 # give a number of this plot
saveDir = '/scratch/palmdata/pp/' + prjname + '/' + jobname + '/'
saveName = varname + '_contour_nz_' + str(plotNo) + '.png'
# plt.savefig(saveDir + saveName, bbox_inches='tight')
plt.show()
