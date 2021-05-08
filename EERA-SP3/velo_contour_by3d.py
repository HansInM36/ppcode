import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/palm')
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import palm_funcs
import matplotlib.pyplot as plt


""" PALM x-y plane """
jobName  = 'EERASP3_1_yw'
dir = "/scratch/palmdata/JOBS/" + jobName
run_no_list = ['.000','.001','.002']

time = []
uSeq = []
vSeq = []

zInd = 10

for i in range(len(run_no_list)):
    run_no = run_no_list[i]
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    tSeq = np.array(input_data.variables['time'][:], dtype=type(input_data.variables['time']))
    tNum = tSeq.size
    
    for tInd in range(tNum):
        t, zSeq, ySeq, xuSeq, uSeq_ = palm_funcs.palm_3d_single_time(dir, jobName, run_no, tInd, 'u')
        t, zSeq, yvSeq, xSeq, vSeq_ = palm_funcs.palm_3d_single_time(dir, jobName, run_no, tInd, 'v')
        
        time.append(t)
        uSeq.append(uSeq_[zInd,:,:])
        vSeq.append(vSeq_[zInd,:,:])

time = np.array(time)     
uSeq = np.array(uSeq)
vSeq = np.array(vSeq)

for tInd in range(time.size):
    fig, axs = plt.subplots(1,1, constrained_layout=False)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    
    ### contour
#    vMin, vMax, vDelta = (-15.0, -0.0, 3.0)
    vMin, vMax, vDelta = (-10.0, -8.0, 0.5)
    cbreso = 100 # resolution of colorbar
    levels = np.linspace(vMin, vMax, cbreso + 1)
    
    CS = axs.contourf(xSeq, ySeq, uSeq[tInd], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    axs.set_xlim([0,15360])
    axs.set_ylim([0,15360])
    axs.set_ylabel('y (m)', fontsize=12)
    axs.set_xlabel('x (m)', fontsize=12)
    axs.text(1.02, 0.5, 'PALM', transform=axs.transAxes, rotation=90, fontsize=12)
    
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    
    axs.set_title('t =' + str(np.round(time[tInd]/3600,1)) + 'h')
    
    saveName = 't' + str(np.round(time[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/u_contour/1_yw'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()







""" PALM x-z plane """
jobName  = 'EERASP3_1_yw_wf'
dir = "/scratch/palmdata/JOBS/" + jobName
run_no_list = ['.000','.001','.002','.003']

time = []
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
        
        time.append(t)
        uSeq.append(uSeq_[:,ySeq.size//2,:])
        vSeq.append(vSeq_[:,ySeq.size//2,:])

time = np.array(time)     
uSeq = np.array(uSeq)
vSeq = np.array(vSeq)

for tInd in range(time.size):
    fig, axs = plt.subplots(1,1, constrained_layout=False)
    fig.set_figwidth(12)
    fig.set_figheight(4)
    
    ### contour
    vMin, vMax, vDelta = (-6.0, 6.0, 2.0)
    cbreso = 100 # resolution of colorbar
    levels = np.linspace(vMin, vMax, cbreso + 1)
    
    CS = axs.contourf(xSeq, zSeq, uSeq[tInd], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    axs.set_xlim([0,15360])
    axs.set_ylim([0,1200])
    axs.set_ylabel('z (m)', fontsize=12)
    axs.set_xlabel('x (m)', fontsize=12)
    axs.text(1.02, 0.5, 'PALM', transform=axs.transAxes, rotation=90, fontsize=12)
    
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    
#    saveName = 't' + str(np.round(time[tInd],1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_bc'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()