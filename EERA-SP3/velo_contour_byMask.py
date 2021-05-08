#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
import pickle
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt

def getInfo_palm(dir, jobName, maskID, run_no, var):
    """ get information of x,y,z,t to decide how much data we should extract """
    input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no + ".nc"
    input_file = Dataset(input_file, "r", format="NETCDF4")
    zName = list(input_file.variables[var].dimensions)[1]
    zSeq = np.array(input_file.variables[zName][:], dtype=type(input_file.variables[zName]))
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(input_file.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_file.variables[yName][:], dtype=type(input_file.variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(input_file.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_file.variables[xName][:], dtype=type(input_file.variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size
    tSeq = np.array(input_file.variables['time'][:], dtype=type(input_file.variables['time']))
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    print('xMin = ', xSeq[0], ', ', 'xMax = ', xSeq[-1], ', ', 'xNum = ', xNum)
    print('yMin = ', ySeq[0], ', ', 'yMax = ', ySeq[-1], ', ', 'yNum = ', yNum)
    print('zMin = ', zSeq[0], ', ', 'zMax = ', zSeq[-1], ', ', 'zNum = ', zNum)
    print('tMin = ', tSeq[0], ', ', 'tMax = ', tSeq[-1], ', ', 'tNum = ', tNum)
    return tSeq, xSeq, ySeq, zSeq

def getData_palm(dir, jobName, maskID, run_no_list, var, tInd, xInd, yInd, zInd):
    """ extract velocity data of specified probe groups """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []

    tInd_start = 0
    list_num = 0
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1

        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][zInd[0]:zInd[1]], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][yInd[0]:yInd[1]], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][xInd[0]:xInd[1]], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(list_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq


""" u contour """
jobName  = 'EERASP3_1'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.000', 'u')

# height
zInd = 6
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.000', '.001', '.002'], 'u', [0,30], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))

vMin, vMax, vDelta = (-15.0, -7.0, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(tSeq.size):
    fig, axs = plt.subplots(figsize=(6,4.8), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    axs.text(0.8, 1.02, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    plt.xlim([2680,12680])
    plt.ylim([2700,12700])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('H = ' + str(np.round(zSeq[0],2)) + 'm' + ', t = ' + str(np.round(tSeq[tInd],2)) + 's')
    saveName = 'H' + str(np.round(zSeq[0],1)) + '_' + 't' + str(np.round(tSeq[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/u_contour/pcr_1/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close('all')


""" u contour """
jobName  = 'EERASP3_2'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.000', 'u')

# height
zInd = 6
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.000', '.001', '.002'], 'u', [0,30], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))

vMin, vMax, vDelta = (7.0, 15.0, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(tSeq.size):
    fig, axs = plt.subplots(figsize=(6,4.8), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    axs.text(0.8, 1.02, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    plt.xlim([2680,12680])
    plt.ylim([2700,12700])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('H = ' + str(np.round(zSeq[0],2)) + 'm' + ', t = ' + str(np.round(tSeq[tInd],2)) + 's')
    saveName = 'H' + str(np.round(zSeq[0],1)) + '_' + 't' + str(np.round(tSeq[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/u_contour/pcr_2/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close('all')


""" u contour """
jobName  = 'EERASP3_1_wf'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.000', 'u')

# height
zInd = 1
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.000', '.001', '.002', '.003'], 'u', [0,30], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))

vMin, vMax, vDelta = (-9.0, -1.0, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(tSeq.size):
    fig, axs = plt.subplots(figsize=(6,4.8), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    axs.text(0.8, 1.02, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    plt.xlim([2680,12680])
    plt.ylim([2700,12700])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('H = ' + str(np.round(zSeq[0],2)) + 'm' + ', t = ' + str(np.round(tSeq[tInd],2)) + 's')
    saveName = 'H' + str(np.round(zSeq[0],1)) + '_' + 't' + str(np.round(tSeq[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/u_contour/scr_1/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close('all')
    

""" u contour """
jobName  = 'EERASP3_2_wf'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.000', 'u')

# height
zInd = 1
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.000', '.001', '.002', '.003'], 'u', [0,30], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))

vMin, vMax, vDelta = (4.0, 12.0, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(tSeq.size):
    fig, axs = plt.subplots(figsize=(6,4.8), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    axs.text(0.8, 1.02, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    plt.xlim([2680,12680])
    plt.ylim([2700,12700])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('H = ' + str(np.round(zSeq[0],2)) + 'm' + ', t = ' + str(np.round(tSeq[tInd],2)) + 's')
    saveName = 'H' + str(np.round(zSeq[0],1)) + '_' + 't' + str(np.round(tSeq[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/u_contour/scr_2/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close('all')