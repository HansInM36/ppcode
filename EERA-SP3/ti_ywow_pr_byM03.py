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

#def getInfo_palm(dir, jobName, maskID, run_no, var):
#    """ get information of x,y,z,t to decide how much data we should extract """
#    input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no + ".nc"
#    input_file = Dataset(input_file, "r", format="NETCDF4")
#    zName = list(input_file.variables[var].dimensions)[1]
#    zSeq = np.array(input_file.variables[zName][:], dtype=type(input_file.variables[zName]))
#    zNum = zSeq.size
#    zSeq = zSeq.astype(float)
#    yName = list(input_file.variables[var].dimensions)[2] # the height name string
#    ySeq = np.array(input_file.variables[yName][:], dtype=type(input_file.variables[yName])) # array of height levels
#    ySeq = ySeq.astype(float)
#    yNum = ySeq.size
#    xName = list(input_file.variables[var].dimensions)[3] # the height name string
#    xSeq = np.array(input_file.variables[xName][:], dtype=type(input_file.variables[xName])) # array of height levels
#    xSeq = xSeq.astype(float)
#    xNum = xSeq.size
#    tSeq = np.array(input_file.variables['time'][:], dtype=type(input_file.variables['time']))
#    tSeq = tSeq.astype(float)
#    tNum = tSeq.size
#    print('xMin = ', xSeq[0], ', ', 'xMax = ', xSeq[-1], ', ', 'xNum = ', xNum)
#    print('yMin = ', ySeq[0], ', ', 'yMax = ', ySeq[-1], ', ', 'yNum = ', yNum)
#    print('zMin = ', zSeq[0], ', ', 'zMax = ', zSeq[-1], ', ', 'zNum = ', zNum)
#    print('tMin = ', tSeq[0], ', ', 'tMax = ', tSeq[-1], ', ', 'tNum = ', tNum)
#    return tSeq, xSeq, ySeq, zSeq

def getInfo_palm(dir, jobName, maskID, run_no_list, var):
    """ extract velocity data of specified probe groups """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        
        tSeq_list.append(tSeq_tmp)

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(len(tSeq_list))], axis=0)
    tSeq = tSeq.astype(float)

    return tSeq, zSeq, ySeq, xSeq

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

    return tSeq, zSeq, ySeq, xSeq, varSeq


def find_nearest_smaller_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be smaller or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] <= v:
        return ind
    else:
        return ind-1

def find_nearest_larger_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be larger or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] > v:
        return ind
    else:
        return ind+1
    
    
""" estimate the TI with masked data """
### (note that this is rough estimate because the C-grid effect is not considered)
""" EERASP3_1 """
jobDir = '/scratch/palmdata/JOBS/EERASP3_1'
jobName  = 'EERASP3_1'
jobDir = prjDir + '/' + jobName
tSeq, zSeq, ySeq, xuSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u')
tSeq, zSeq, yvSeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v')
tSeq, zwSeq, ySeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w')

t_plot = np.arange(1,13) * 3600 # times of profiles that are going to be plotted

TIu_ref = np.zeros((t_plot.size, zSeq.size-1))
TIv_ref = np.zeros((t_plot.size, zSeq.size-1))
TIw_ref = np.zeros((t_plot.size, zSeq.size-1))

for zInd in range(1,zSeq.size): # exclude the zero height

    u_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    v_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    
    for i in range(t_plot.size):
        tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
        tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])

        t = tSeq[tInd_0:tInd_1+1]
        u = u_org[4][tInd_0:tInd_1+1]
        v = v_org[4][tInd_0:tInd_1+1]
        w = w_org[4][tInd_0:tInd_1+1]
        
        u_ave = np.average(u, axis=0)
        v_ave = np.average(v, axis=0)
        uv_ave = np.power(np.power(u_ave,2) + np.power(v_ave,2), 0.5)
        
        sgm2u = np.zeros(u_ave.shape)
        sgm2v = np.zeros(u_ave.shape)
        sgm2w = np.zeros(u_ave.shape)
        
        # calculate the 3d TI at all grid points
        for yInd in range(sgm2u.shape[0]):
            for xInd in range(sgm2u.shape[1]):
                u_ = u[:,0,yInd,xInd] - u_ave[0,yInd,xInd]
                v_ = v[:,0,yInd,xInd] - v_ave[0,yInd,xInd]
                w_ = w[:,0,yInd,xInd] - 0
                # detrend
                u_ = funcs.detrend(t, u_)
                v_ = funcs.detrend(t, v_)
                w_ = funcs.detrend(t, w_)
                # variance
                sgm2u[yInd,xInd] = np.var(u_)
                sgm2v[yInd,xInd] = np.var(v_)
                sgm2w[yInd,xInd] = np.var(w_)
        
        TIu_ref[i,zInd-1] = np.mean(np.power(sgm2u,0.5)/uv_ave)
        TIv_ref[i,zInd-1] = np.mean(np.power(sgm2v,0.5)/uv_ave)
        TIw_ref[i,zInd-1] = np.mean(np.power(sgm2w,0.5)/uv_ave)
        
        
""" EERASP3_1_yw """
jobDir = '/scratch/palmdata/JOBS/EERASP3_1_yw'
jobName  = 'EERASP3_1_yw'
jobDir = prjDir + '/' + jobName
tSeq, zSeq, ySeq, xuSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u')
tSeq, zSeq, yvSeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v')
tSeq, zwSeq, ySeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w')

t_plot = np.arange(1,13) * 3600 # times of profiles that are going to be plotted

TIu_yw = np.zeros((t_plot.size, zSeq.size-1))
TIv_yw = np.zeros((t_plot.size, zSeq.size-1))
TIw_yw = np.zeros((t_plot.size, zSeq.size-1))

for zInd in range(1,zSeq.size): # exclude the zero height

    u_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    v_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    
    for i in range(t_plot.size):
        tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
        tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])

        t = tSeq[tInd_0:tInd_1+1]
        u = u_org[4][tInd_0:tInd_1+1]
        v = v_org[4][tInd_0:tInd_1+1]
        w = w_org[4][tInd_0:tInd_1+1]
        
        u_ave = np.average(u, axis=0)
        v_ave = np.average(v, axis=0)
        uv_ave = np.power(np.power(u_ave,2) + np.power(v_ave,2), 0.5)
        
        sgm2u = np.zeros(u_ave.shape)
        sgm2v = np.zeros(u_ave.shape)
        sgm2w = np.zeros(u_ave.shape)
        
        # calculate the 3d TI at all grid points
        for yInd in range(sgm2u.shape[0]):
            for xInd in range(sgm2u.shape[1]):
                u_ = u[:,0,yInd,xInd] - u_ave[0,yInd,xInd]
                v_ = v[:,0,yInd,xInd] - v_ave[0,yInd,xInd]
                w_ = w[:,0,yInd,xInd] - 0
                # detrend
                u_ = funcs.detrend(t, u_)
                v_ = funcs.detrend(t, v_)
                w_ = funcs.detrend(t, w_)
                # variance
                sgm2u[yInd,xInd] = np.var(u_)
                sgm2v[yInd,xInd] = np.var(v_)
                sgm2w[yInd,xInd] = np.var(w_)
        
        TIu_yw[i,zInd-1] = np.mean(np.power(sgm2u,0.5)/uv_ave)
        TIv_yw[i,zInd-1] = np.mean(np.power(sgm2v,0.5)/uv_ave)
        TIw_yw[i,zInd-1] = np.mean(np.power(sgm2w,0.5)/uv_ave)        

""" EERASP3_1_ow """
jobDir = '/scratch/palmdata/JOBS/EERASP3_1_ow'
jobName  = 'EERASP3_1_ow'
jobDir = prjDir + '/' + jobName
tSeq, zSeq, ySeq, xuSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u')
tSeq, zSeq, yvSeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v')
tSeq, zwSeq, ySeq, xSeq = getInfo_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w')

t_plot = np.arange(1,13) * 3600 # times of profiles that are going to be plotted

TIu_ow = np.zeros((t_plot.size, zSeq.size-1))
TIv_ow = np.zeros((t_plot.size, zSeq.size-1))
TIw_ow = np.zeros((t_plot.size, zSeq.size-1))

for zInd in range(1,zSeq.size): # exclude the zero height

    u_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'u', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    v_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'v', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org = getData_palm(jobDir, jobName, 'M03', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    
    for i in range(t_plot.size):
        tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
        tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])

        t = tSeq[tInd_0:tInd_1+1]
        u = u_org[4][tInd_0:tInd_1+1]
        v = v_org[4][tInd_0:tInd_1+1]
        w = w_org[4][tInd_0:tInd_1+1]
        
        u_ave = np.average(u, axis=0)
        v_ave = np.average(v, axis=0)
        uv_ave = np.power(np.power(u_ave,2) + np.power(v_ave,2), 0.5)
        
        sgm2u = np.zeros(u_ave.shape)
        sgm2v = np.zeros(u_ave.shape)
        sgm2w = np.zeros(u_ave.shape)
        
        # calculate the 3d TI at all grid points
        for yInd in range(sgm2u.shape[0]):
            for xInd in range(sgm2u.shape[1]):
                u_ = u[:,0,yInd,xInd] - u_ave[0,yInd,xInd]
                v_ = v[:,0,yInd,xInd] - v_ave[0,yInd,xInd]
                w_ = w[:,0,yInd,xInd] - 0
                # detrend
                u_ = funcs.detrend(t, u_)
                v_ = funcs.detrend(t, v_)
                w_ = funcs.detrend(t, w_)
                # variance
                sgm2u[yInd,xInd] = np.var(u_)
                sgm2v[yInd,xInd] = np.var(v_)
                sgm2w[yInd,xInd] = np.var(w_)
        
        TIu_ow[i,zInd-1] = np.mean(np.power(sgm2u,0.5)/uv_ave)
        TIv_ow[i,zInd-1] = np.mean(np.power(sgm2v,0.5)/uv_ave)
        TIw_ow[i,zInd-1] = np.mean(np.power(sgm2w,0.5)/uv_ave)


""" TI cmp """
for i in range(t_plot.size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
#    plt.plot(TIu_ref[i]*100, zSeq[1:], label='ref', linestyle='-', linewidth=1.0, color='k')
    plt.plot(TIu_yw[i]*100, zSeq[1:], label='yw', marker='', linestyle='-', linewidth=1.0, color='b')
    plt.plot(TIu_ow[i]*100, zSeq[1:], label='ow', marker='', linestyle='-', linewidth=1.0, color='r')
    plt.xlabel(r"$\mathrm{TI_u}$ (%)")
    plt.ylabel('z (m)')
    xaxis_min = 0
    xaxis_max = 3
    xaxis_d = 0.5
    yaxis_min = 0
    yaxis_max = 200.0
    yaxis_d = 20.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    plt.legend(bbox_to_anchor=(0.66,0.8), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t =' + str(np.round(t_plot[i]/3600,1)) + 'h') 
    fig.tight_layout() # adjust the layout
    saveName = 'TIu_pr_t' + str(np.round(t_plot[i]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/TIu_pr/1st'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()



""" TI evo of one case """
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,t_plot.size))

for i in range(t_plot.size):
#    # add 0 to the zero height
#    zero = np.zeros(1)
#    v_ = np.concatenate((zero, varplotList[i]))
    plt.plot(TIw[i]*100, zSeq[1:], label='t = ' + str(int(t_plot[i])/3600) + 'h', linewidth=1.0, color=colors[i])
plt.xlabel(r"$\mathrm{TI_u}$ (%)")
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 5
xaxis_d = 1
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
plt.show()









