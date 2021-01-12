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
def getSliceData_Nz_sowfa(dir, jobName, slice, var, varD, trs_para, tInd, xInd, yInd):
    """ extract velocity data of specified slice_Nz """
    readDir = dir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data = pickle.load(fr)
    fr.close()
    slc = sdc.Slice(data, 2); del data # 2 means z-axis
    tSeq = slc.data['time']
    tSeq = tSeq[tInd[0]:tInd[1]]
    H = slc.N_location # height of this plane

    O, alpha = trs_para[0], trs_para[1]
    varSeq = []
    for t_ind in range(tInd[0],tInd[1]):
        print('processing: ' + str(t_ind) + ' ...')
        tmp = slc.data[var][t_ind]
        tmp = funcs.trs(tmp,O,alpha)
        tmp1 = slc.meshITP_Nz(xInd, yInd, tmp[:,varD], method_='linear')
        xSeq, ySeq, varSeq_ = tmp1[0], tmp1[1], tmp1[2]
        varSeq.append(varSeq_)
    varSeq = np.arrray(varSeq)
    return tSeq, xSeq, ySeq, H, varSeq

# dir, jobName, slice, var, varD, trs_para, tInd, xInd, yInd = ppDir, jobName, 'Nz0', 'U', 0, ((0,0,0),30), (0,150), (0,2560,256), (0,2560,256)


""" animation for deepwind_gs10 (SOWFA) """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq, xSeq, ySeq, H, varSeq = getSliceData_Nz_sowfa(ppDir, jobName, 'Nz2', 'U', 0, ((0,0,0),30), (0,150), (0,2560,256), (0,2560,256))
t0 = tSeq[0]

vMin, vMax, vDelta = (-2, 2, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

# transform coors of prbg
prbg0 = np.vstack((780 + 20*np.arange(0,51), np.array([1280 for i in range(51)]), np.array([0 for i in range(51)])))
prbg0 = funcs.trs(prbg0.T, (1280,1280,0), -30); prbg0[:,0] += 1280; prbg0[:,1] += 1280;
prbg1 = np.vstack((np.array([1280 for i in range(51)]), 780 + 20*np.arange(0,51), np.array([0 for i in range(51)])))
prbg1 = funcs.trs(prbg1.T, (1280,1280,0), -30); prbg1[:,0] += 1280; prbg1[:,1] += 1280;


for tInd in range(0,150,1):
    fig, axs = plt.subplots(figsize=(4.5,4.5), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd]
    v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    plt.scatter(prbg0[:,0], prbg0[:,1], 1, marker='o', color='k')
    plt.scatter(prbg1[:,0], prbg1[:,1], 1, marker='o', color='k')
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,2560])
    plt.ylim([0,2560])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/projects/deepwind/photo/animation/Nz_sowfa'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')



""" animation for deepwind_gs10 (PALM) """
jobName  = 'deepwind_gs10'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.022', 'u')
t0 = tSeq[0]
# height
zInd = 5
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.022','.023'], 'u', (0,300), (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
print(np.min(varSeq-varSeq.mean()),np.max(varSeq-varSeq.mean())) # find min and max


""" animation for deepwind_NBL (PALM) """
jobName  = 'deepwind_NBL'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.002', 'u')
t0 = tSeq[0]
# height
zInd = 4
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M01', ['.002'], 'u', (0,300), (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
print(np.min(varSeq-varSeq.mean()),np.max(varSeq-varSeq.mean())) # find min and max


vMin, vMax, vDelta = (-2, 2, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

prbg0 = [list(780 + 20*np.arange(0,51)), [1280 for i in range(51)]]
prbg1 = [[1280 for i in range(51)], list(780 + 20*np.arange(0,51))]

for tInd in range(0,150,1):
    fig, axs = plt.subplots(figsize=(4.5,4.5), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    plt.scatter(prbg0[0]+prbg1[0], prbg0[1]+prbg1[1], 1, marker='o', color='k')
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,2560])
    plt.ylim([0,2560])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/projects/deepwind/photo/animation/Nz_palm'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')


""" animation for pcr_u12_nbl (PALM) """
jobName  = 'pcr_u12_nbl'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M04', '.007', 'u')
t0 = tSeq[0]
# height
zInd = 0
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(jobDir, jobName, 'M04', ['.007'], 'u', [0,150], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
print(np.min(varSeq-varSeq.mean()),np.max(varSeq-varSeq.mean())) # find min and max

vMin, vMax, vDelta = (-2, 2, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

prbg0 = [list(780 + 20*np.arange(0,51)), [1280 for i in range(51)]]
prbg1 = [[1280 for i in range(51)], list(780 + 20*np.arange(0,51))]

for tInd in range(0,150,1):
    fig, axs = plt.subplots(figsize=(4.5,4.5), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = varSeq[tInd,0]
    v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    plt.scatter(prbg0[0]+prbg1[0], prbg0[1]+prbg1[1], 1, marker='o', color='k')
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,2560])
    plt.ylim([0,2560])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/projects/deepwind/animation/' + jobName
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')
