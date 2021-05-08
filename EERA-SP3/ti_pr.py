import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt

def getData_palm(dir, jobName, maskID, run_no_list, var):
    """ extract velocity data of specified probe groups """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'WRFPALM_20150701'
dir = prjDir + '/' + jobName
data_0 = getData_palm(dir, jobName, 'M05', ['.000','.001','.002','.003','.004','.005','.006','.007'], 'u')


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'WRFPALM_20150701_nostg'
dir = prjDir + '/' + jobName
data_1 = getData_palm(dir, jobName, 'M05', ['.000','.001','.002','.003'], 'u')


tList = [3,6,9,12]

for t in tList:
    inds_0 = np.where((data_0[0] > (t-1)*3600) & (data_0[0] <= t*3600))[0]
    u0 = data_0[4][inds_0,:,0,0]
    
    inds_1 = np.where((data_1[0] > (t-1)*3600) & (data_1[0] <= t*3600))[0]
    u1 = data_1[4][inds_1,:,0,0]
    
    umean0 = np.abs(np.mean(u0, axis=0))
    umean1 = np.abs(np.mean(u1, axis=0))
    
    for zInd in range(data_0[3].size):
        u0[:,zInd] = funcs.detrend(data_0[0][inds_0], u0[:,zInd])
        u1[:,zInd] = funcs.detrend(data_1[0][inds_1], u1[:,zInd])

    ti0 = np.sqrt(np.var(u0, axis=0)) / umean0
    ti1 = np.sqrt(np.var(u1, axis=0)) / umean1
    
    fig, ax = plt.subplots(figsize=(3.0,4.5))
    plt.plot(ti0*100, data_0[3], label='ref', linestyle='-', linewidth=1.0, color='k')
    plt.plot(ti1*100, data_1[3], label='nostg', linestyle='--', linewidth=1.0, color='k')
    plt.xlabel('TI (%)', fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 0.0
    xaxis_max = 5.0
    xaxis_d = 1.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(np.round(t*3600,1)) + 's')
    fig.tight_layout() # adjust the layout
    saveName = 'pr_ti_t' + str(np.round(t,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/ti_pr'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()   
    
    
    