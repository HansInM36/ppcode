import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

def palm_3d_single(dir, jobName, run_no, tInd, var):
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    t = np.array(input_data.variables['time'][tInd], dtype=float)
    varSeq = np.array(input_data.variables[var][tInd], dtype=float)
    
    # extract the values of all dimensions of the var
    zName = list(input_data.variables[var].dimensions)[1] # the height name string
    zSeq = np.array(input_data.variables[zName][:], dtype=float) # array of height levels
    yName = list(input_data.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_data.variables[yName][:], dtype=float) # array of height levels
    xName = list(input_data.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_data.variables[xName][:], dtype=float) # array of height levels
    return t, zSeq, ySeq, xSeq, varSeq


""" WRF """
readDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/INPUT'
readName = "WRFPALM_20150701_dynamic"

nx, ny, nz = 256, 256, 96
dx, dy, dz = 10, 10, 10

wrf = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(wrf.dimensions)
varlist = list(wrf.variables)

TIME = np.array(wrf.variables['time'][:], dtype=float)
Y = np.array(wrf.variables['y'][:], dtype=float)
Z = np.array(wrf.variables['z'][:], dtype=float)
left_U = np.array(wrf.variables['ls_forcing_left_u'][:], dtype=float)
left_V = np.array(wrf.variables['ls_forcing_left_v'][:], dtype=float)


""" PALM """
jobName  = 'WRFPALM_20150701'
dir = "/scratch/palmdata/JOBS/" + jobName
run_no_list = ['.000','.001','.002','.003','.004','.005','.006','.007']

uSeq = []
vSeq = []

for i in range(len(run_no_list)):
    run_no = run_no_list[i]
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    tSeq = np.array(input_data.variables['time'][:], dtype=type(input_data.variables['time']))
    tNum = tSeq.size
    
    for tInd in range(tNum):
        t, zSeq, ySeq, xuSeq, uSeq_ = palm_3d_single(dir, jobName, run_no, tInd, 'u')
        t, zSeq, yvSeq, xSeq, vSeq_ = palm_3d_single(dir, jobName, run_no, tInd, 'v')
        
        uSeq.append(uSeq_[:,:,0])
        vSeq.append(vSeq_[:,:,0])
        
uSeq = np.array(uSeq)
vSeq = np.array(vSeq)


""" left boundary plane """
for tInd in range(1,TIME.size-1):
    fig, axs = plt.subplots(2,1, constrained_layout=False)
    fig.set_figwidth(12)
    fig.set_figheight(4)
    
    ### contour
    vMin, vMax, vDelta = (-14.0, -2.0, 2.0)
    cbreso = 100 # resolution of colorbar
    levels = np.linspace(vMin, vMax, cbreso + 1)
    
    CS = axs[0].contourf(Y, Z, left_U[tInd], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    axs[0].set_xlim([0,15360])
    axs[0].set_ylim([0,1200])
    axs[0].set_ylabel('y (m)', fontsize=12)
    #axs[0].set_xlabel('x (m)', fontsize=12)
    axs[0].text(1.02, 0.5, 'WRF', transform=axs[0].transAxes, rotation=90, fontsize=12)
    
    CS = axs[1].contourf(ySeq, zSeq, uSeq[tInd], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    axs[1].set_xlim([0,15360])
    axs[1].set_ylim([0,1200])
    axs[1].set_ylabel('y (m)', fontsize=12)
    axs[1].set_xlabel('x (m)', fontsize=12)
    axs[1].text(1.02, 0.5, 'PALM', transform=axs[1].transAxes, rotation=90, fontsize=12)
    
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    
    saveName = 't' + str(np.round(TIME[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_bc'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()