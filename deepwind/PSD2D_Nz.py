import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
import pickle
import scipy.signal
import scipy.fft
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

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
        tmp1 = slc.meshITP_Nz(xInd, yInd, tmp[:,varD], method_='cubic')
        xSeq, ySeq, varSeq_ = tmp1[0], tmp1[1], tmp1[2]
        varSeq.append(varSeq_)
    varSeq = np.array(varSeq)
    return tSeq, xSeq, ySeq, H, varSeq

def PSD2D(tSeq, dy, dx, varSeq):
    psd2d = []
    for tInd in range(tSeq.size):
        tmp = varSeq[tInd] - varSeq[tInd].mean()
        psd2d_, kx, ky = funcs.PSD2D(tmp, dx, dy)
        psd2d.append(psd2d_)
    psd2d = np.average(np.array(psd2d),axis=0)
    return psd2d, kx, ky


""" 2d PSD (PALM) """
jobName  = 'deepwind_NBL'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M01', '.002', 'u')
# height
zInd = 4
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0 = getData_palm(jobDir, jobName, 'M01', ['.002'], 'u', (0,10), (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0 = getData_palm(jobDir, jobName, 'M01', ['.002'], 'v', (0,10), (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
uvSeq_0 = np.power(np.power(uSeq_0,2) + np.power(vSeq_0,2), 0.5)
uvSeq_0 = uvSeq_0 - np.mean(uvSeq_0)
psd2d_0, kx_0, ky_0 = PSD2D(tSeq_0, 10, 10, uvSeq_0[:,0,:,:])

""" 2d PSD (SOWFA) """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, H, uSeq_1 = getSliceData_Nz_sowfa(ppDir, jobName, 'Nz2', 'U', 0, ((0,0,0),30), (0,10), (0,2560,1280), (0,2560,1280))
tSeq_1, xSeq_1, ySeq_1, H, vSeq_1 = getSliceData_Nz_sowfa(ppDir, jobName, 'Nz2', 'U', 1, ((0,0,0),30), (0,10), (0,2560,1280), (0,2560,1280))
uvSeq_1 = np.power(np.power(uSeq_1,2) + np.power(vSeq_1,2), 0.5)
uvSeq_1 = uvSeq_1 - np.mean(uvSeq_1)
psd2d_1, kx_1, ky_1 = PSD2D(tSeq_1, 10, 10, uvSeq_1)

### plot
fig, axs = plt.subplots(figsize=(6.6,6), constrained_layout=False)
vMin, vMax, vN = (-6,4,11)
# cbreso = 100 # resolution of colorbar
pwlevels = np.linspace(vMin,vMax,vN*10)
levels = np.power(10., pwlevels)
x_ = np.copy(kx_1)
y_ = np.copy(ky_1)
v_ = np.copy(psd2d_1)
v_[np.where(v_ < np.power(10., vMin))] = np.power(10., vMin)
v_[np.where(v_ > np.power(10., vMax))] = np.power(10., vMax)
CS = axs.contourf(x_, y_, v_, levels=levels, norm=clrs.LogNorm(), cmap='jet')
cbartickList = np.power(10., np.linspace(vMin,vMax,vN))
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)

cbar.ax.set_ylabel(r"$\mathrm{S_u}$", fontsize=12)
# plt.xlim([0,2560])
# plt.ylim([0,2560])
plt.ylabel(r'$\mathrm{k_y}$ (1/m)', fontsize=12)
plt.xlabel(r'$\mathrm{k_x}$ (1/m)', fontsize=12)
plt.title('')
# saveName = "%.4d" % tInd + '.png'
# saveDir = '/scratch/projects/deepwind/photo/animation/Nz_sowfa'
# if not os.path.exists(saveDir):
#     os.makedirs(saveDir)
# plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close('all')
