#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
from scipy.interpolate import interp1d
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
    return tSeq, xSeq, ySeq, H, varSeq


""" snapshot of u'w' """
jobName  = 'wwinta_gs20_regular_hws'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.001', 'u')
t0 = tSeq[0]

uStar = 0.4863

# height
zInd = 1

tSeq, xuSeq, yuSeq, zuSeq, uSeq = getData_palm(jobDir, jobName, 'M01', ['.001'], 'u', [0,599], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
tSeq, xwSeq, ywSeq, zwSeq, wSeq = getData_palm(jobDir, jobName, 'M01', ['.001'], 'w', [0,599], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))

# compute u'w'

xSeq_, ySeq_, zSeq_ = xwSeq, ywSeq, zuSeq # positions (scalar position) of cell centers 
uSeq_ = np.zeros((tSeq.size, zSeq_.size, ySeq_.size, xSeq_.size))
wSeq_ = np.zeros((tSeq.size, zSeq_.size, ySeq_.size, xSeq_.size))

f_itp_u = interp1d(xuSeq, uSeq, axis=3, kind='linear', fill_value='extrapolate')
#f_itp_w = interp1d(zwSeq, wSeq, axis=1, kind='linear', fill_value='extrapolate') # only one z value, so cannot interpolate 

for xInd in range(xSeq_.size):
    uSeq_[:,:,:,xInd] = f_itp_u(xSeq[xInd])

#for zInd in range(zSeq_.size):
#    wSeq_[:,zInd,:,:] = f_itp_w(zSeq[zInd])
wSeq_ = wSeq/2 # we just divide the data by 2 for the first layer grid as the approximation of data at scalar positions

uSeq_enav = np.average(uSeq_,axis=0) 
wSeq_enav = np.average(wSeq_,axis=0)

uSeq_ -= uSeq_enav
wSeq_ -= wSeq_enav
uwSeq_ = uSeq_*wSeq_

# plot
tInd = 150
fig, axs = plt.subplots(figsize=(4.5,4.5), constrained_layout=False)
vMin, vMax, vDelta = (-1.8, 1.2, 0.5)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
x_ = xSeq_
y_ = ySeq_
v_ = uwSeq_[tInd,0]/uStar/uStar
v_[np.where(v_ < vMin)] = vMin
v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
axs.text(0.0, 1.02, 'h = ' + str(np.round(zSeq_[0],2)) + 'm', transform=axs.transAxes, fontsize=12)
axs.text(0.8, 1.02, 't = ' + str(np.round(tSeq[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
axs.text(0.24, 1.08, jobName, transform=axs.transAxes, fontsize=12)
cbar.ax.set_ylabel(r"$u'w'/u_*^2$", fontsize=12)
plt.xlim([0,1280])
plt.ylim([0,1280])
plt.ylabel('y (m)', fontsize=12)
plt.xlabel('x (m)', fontsize=12)
#plt.title(jobName)
saveName = jobName + '_uwflux.png'
saveDir = '/scratch/projects/wwinta/photo/uwflux'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close('all')
