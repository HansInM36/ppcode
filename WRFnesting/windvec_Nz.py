import os
import sys
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
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



jobName  = 'WRFPALM_20150701'
dir = "/scratch/palmdata/JOBS/" + jobName
run_no_list = ['.000','.001','.002','.003','.004','.005','.006','.007']

zInd = 11

for i in range(len(run_no_list)):
    run_no = run_no_list[i]
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    tSeq = np.array(input_data.variables['time'][:], dtype=type(input_data.variables['time']))
    tNum = tSeq.size
    
    for tInd in range(tNum):
        t, zSeq, ySeq, xuSeq, uSeq = palm_3d_single(dir, jobName, run_no, tInd, 'u')
        t, zSeq, yvSeq, xSeq, vSeq = palm_3d_single(dir, jobName, run_no, tInd, 'v')
        
        fu = interp2d(xuSeq, ySeq, uSeq[zInd], kind='linear')
        fv = interp2d(xSeq, yvSeq, vSeq[zInd], kind='linear')
        uSeq_ = fu(xSeq, ySeq)
        vSeq_ = fv(xSeq, ySeq)
        uvSeq_ = np.sqrt(np.power(uSeq_,2) + np.power(vSeq_,2))
        
        xx, yy = np.meshgrid(xSeq, ySeq)
        
        fig, ax = plt.subplots(figsize=(7.0,6.0))
        ### contour
        vMin, vMax, vDelta = (8.0, 16.0, 1.0)
        cbreso = 100 # resolution of colorbar
        levels = np.linspace(vMin, vMax, cbreso + 1)
        CS = ax.contourf(xSeq, ySeq, uvSeq_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
        cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
        cbar = plt.colorbar(CS, ax=ax, orientation='vertical', ticks=cbartickList, fraction=.1)
        cbar.ax.set_ylabel(r"$\mathrm{|u_h|}$" + ' (m/s)', fontsize=12)
        plt.xlim([0,15360])
        plt.ylim([0,15360])
        plt.ylabel('y (m)', fontsize=12)
        plt.xlabel('x (m)', fontsize=12)
        ### vector field
        spc = 24
        plt.quiver(xx[::spc,::spc], yy[::spc,::spc], uSeq_[::spc,::spc], vSeq_[::spc,::spc], units='width')
        saveName = 'H' + str(np.round(zSeq[zInd],1)) + '_' + 't' + str(np.round(t,1)) + '.png'
        saveDir = '/scratch/projects/EERA-SP3/photo/windvec_Nz'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
        plt.show()
        plt.close()