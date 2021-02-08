#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
import os
import sys
import pickle
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm


def velo_pr_palm(dir, jobName, run_no_list, var):
    """ extract horizontal average of velocity at various times and heights """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # dimensions = list(nc_file_list[0].dimensions)
    # vars = list(nc_file_list[0].variables)
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, zSeq, varSeq
def velo_pr_ave(tplot_para, tSeq, tDelta, zNum, varSeq):
    """ calculate temporally averaged horizontal average of velocity at various times and heights """
    ave_itv = tplot_para[0]
    tplot_start = tplot_para[1]
    tplot_end = tplot_para[2]
    tplot_delta = tplot_para[3]
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

    # compute the averaged velocity at a certain time and height
    varplotList = []
    for tplot in tplotList:
        varplot = np.zeros(zNum)
        for zInd in range(zNum):
            f = interp1d(tSeq, varSeq[:,zInd], kind='linear', fill_value='extrapolate')
            tplot_tmp = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
            varplot[zInd] = f(tplot_tmp).mean()
        varplotList.append(varplot)
    t_seq = np.array(tplotList)
    varSeq = np.array(varplotList)
    return t_seq, varSeq
def single_plot(varSeq, zSeq):
    """ single velo_pr plot """
    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(varSeq, zSeq, linewidth=1.0, color='k')
    plt.ylabel('z (m)')
    xaxis_min = 0
    xaxis_max = 12.0
    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 1000.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    # plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('')
    fig.tight_layout() # adjust the layout
    plt.show()
def ITP(varSeq, zSeq, z):
    f = interp1d(zSeq, varSeq, kind='linear')
    return f(z)
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

### read netcdf data
readDir = '/scratch/palmdata/JOBS/WRFnesting_stg/INPUT'
readName = "WRFnesting_stg_dynamic"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")
tSeq = data.variables['time'][:]
zSeq = data.variables['z'][:]
ySeq = data.variables['y'][:]
xSeq = data.variables['x'][:]

init_atmosphere_pt = np.array(data.variables['init_atmosphere_pt'])
init_atmosphere_qv = np.array(data.variables['init_atmosphere_qv'])

init_atmosphere_u = np.array(data.variables['init_atmosphere_u'])
init_atmosphere_v = np.array(data.variables['init_atmosphere_v'])
init_atmosphere_w = np.array(data.variables['init_atmosphere_w'])

ls_forcing_ug = np.array(data.variables['ls_forcing_ug'])
ls_forcing_vg = np.array(data.variables['ls_forcing_vg'])

ls_forcing_left_pt = np.array(data.variables['ls_forcing_left_pt'])
ls_forcing_right_pt = np.array(data.variables['ls_forcing_right_pt'])
ls_forcing_south_pt = np.array(data.variables['ls_forcing_south_pt'])
ls_forcing_north_pt = np.array(data.variables['ls_forcing_north_pt'])
ls_forcing_top_pt = np.array(data.variables['ls_forcing_top_pt'])

ls_forcing_left_qv = np.array(data.variables['ls_forcing_left_qv'])
ls_forcing_right_qv = np.array(data.variables['ls_forcing_right_qv'])
ls_forcing_south_qv = np.array(data.variables['ls_forcing_south_qv'])
ls_forcing_north_qv = np.array(data.variables['ls_forcing_north_qv'])
ls_forcing_top_qv = np.array(data.variables['ls_forcing_top_qv'])

ls_forcing_left_u = np.array(data.variables['ls_forcing_left_u'])
ls_forcing_right_u = np.array(data.variables['ls_forcing_right_u'])
ls_forcing_south_u = np.array(data.variables['ls_forcing_south_u'])
ls_forcing_north_u = np.array(data.variables['ls_forcing_north_u'])
ls_forcing_top_u = np.array(data.variables['ls_forcing_top_u'])

ls_forcing_left_v = np.array(data.variables['ls_forcing_left_v'])
ls_forcing_right_v = np.array(data.variables['ls_forcing_right_v'])
ls_forcing_south_v = np.array(data.variables['ls_forcing_south_v'])
ls_forcing_north_v = np.array(data.variables['ls_forcing_north_v'])
ls_forcing_top_v = np.array(data.variables['ls_forcing_top_v'])

ls_forcing_left_w = np.array(data.variables['ls_forcing_left_w'])
ls_forcing_right_w = np.array(data.variables['ls_forcing_right_w'])
ls_forcing_south_w = np.array(data.variables['ls_forcing_south_w'])
ls_forcing_north_w = np.array(data.variables['ls_forcing_north_w'])
ls_forcing_top_w = np.array(data.variables['ls_forcing_top_w'])

data.close()


prjDir = '/scratch/palmdata/JOBS'
jobName_0  = 'WRFnesting'
dir_0 = prjDir + '/' + jobName_0
tSeq_0, zSeq_0, uSeq_0 = velo_pr_palm(dir_0, jobName_0, ['.000'], 'u')

prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'WRFnesting_stg'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, zSeq_1, uSeq_1 = velo_pr_palm(dir_1, jobName_1, ['.000'], 'u')
tSeq_1, zSeq_1, ptSeq_1 = velo_pr_palm(dir_1, jobName_1, ['.000'], 'theta')

""" u profile of stationary flow """
fig, ax = plt.subplots(figsize=(6,4.5))
colors = plt.cm.jet(np.linspace(0,1,tSeq_1.size))

for i in range(tSeq_1.size):
    plt.plot(uSeq_1[i], zSeq_1, label='t = ' + str(int(tSeq_1[i])) + 's', linewidth=1.0, linestyle='-', color=colors[i])

## profile from WRF
uSeq_init = np.average(init_atmosphere_u.reshape(96,256*255),axis=1)
uSeq_l = np.average(ls_forcing_left_u[0],axis=1)
uSeq_r = np.average(ls_forcing_right_u[0],axis=1)
plt.plot(uSeq_init, zSeq, label='WRF-init', linewidth=1.0, linestyle='-', color='k')
plt.plot(uSeq_l, zSeq, label='WRF-forcing-left', linewidth=1.0, linestyle='--', color='k')
plt.plot(uSeq_r, zSeq, label='WRF-forcing-right', linewidth=1.0, linestyle=':', color='k')

plt.xlabel(r"$\overline{\mathrm{\overline{u}}}$ (m/s)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = -10.0
xaxis_max = 0.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1200.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'u' + '_pr_1.png'
saveDir = '/scratch/palmdata/pp/WRFnesting_stg/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()


""" temperature profile of stationary flow """
fig, ax = plt.subplots(figsize=(6,4.5))
colors = plt.cm.jet(np.linspace(0,1,tSeq_1.size))

for i in range(0,tSeq_1.size):
    plt.plot(ptSeq_1[i], zSeq_1, label='t = ' + str(int(tSeq_1[i])) + 's', linewidth=1.0, linestyle='-', color=colors[i])

## profile from WRF
ptSeq_init = np.average(init_atmosphere_pt.reshape(96,256*256),axis=1)
ptSeq_l = np.average(ls_forcing_left_pt[0],axis=1)
ptSeq_r = np.average(ls_forcing_right_pt[0],axis=1)
plt.plot(ptSeq_init, zSeq, label='WRF-init', linewidth=1.0, linestyle='-', color='k')
plt.plot(ptSeq_l, zSeq, label='WRF-forcing-left', linewidth=1.0, linestyle='--', color='k')
plt.plot(ptSeq_r, zSeq, label='WRF-forcing-right', linewidth=1.0, linestyle=':', color='k')

plt.xlabel(r"$\overline{\mathrm{\theta}}$ (K)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 285.0
xaxis_max = 305.0
xaxis_d = 5.0
yaxis_min = 0.0
yaxis_max = 1200.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'theta' + '_pr_1.png'
saveDir = '/scratch/palmdata/pp/WRFnesting_stg/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()



""" U_contour in horizontal plane """

### U_contour from WRF
vMin, vMax, vDelta = (-5.00, -4.40, 0.1) # h = 15m
# vMin, vMax, vDelta = (-6.4, -6.1, 0.05) # h = 95m
# vMin, vMax, vDelta = (-6.88, -6.64, 0.04) # h = 175m
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
fig, axs = plt.subplots(figsize=(5.5,5.0), constrained_layout=False)
x_ = np.copy(xSeq)
y_ = np.copy(ySeq)
v_ = np.copy(init_atmosphere_u[17]) # 1,9,17
# v_ -= v_.mean()
# v_[np.where(v_ < vMin)] = vMin
# v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(x_[:-1], y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{U}$" + ' (m/s)', fontsize=12)
plt.xlim([0,2560])
plt.ylim([0,2560])
plt.ylabel('y (m)', fontsize=12)
plt.xlabel('x (m)', fontsize=12)
plt.title('')
saveName = 'U_contour_h175.png'
saveDir = '/scratch/palmdata/pp/WRFnesting_stg/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close('all')



jobName  = 'WRFnesting_stg'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_0, xSeq_0, ySeq_0, zSeq_0 = getInfo_palm(jobDir, jobName, 'M01', '.001', 'u')
t0 = tSeq_0[0]
# height
zInd = 2
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0 = getData_palm(jobDir, jobName, 'M01', ['.001'], 'u', [0,60], (0,xSeq_0.size), (0,ySeq_0.size), (zInd,zInd+1))
print(np.min(varSeq_0-varSeq_0.mean()),np.max(varSeq_0-varSeq_0.mean())) # find min and max


# vMin, vMax, vDelta = (-6.20, -3.40, 0.4)
# vMin, vMax, vDelta = (-1.6, 1.6, 0.4) # 15m
vMin, vMax, vDelta = (-0.18, 0.14, 0.04) # 95m
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(0,60,1):
    fig, axs = plt.subplots(figsize=(4.5,4.5), constrained_layout=False)
    x_ = xSeq_0
    y_ = ySeq_0
    v_ = np.copy(varSeq_0[tInd,0])
    v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq_0[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,2560])
    plt.ylim([0,2560])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/palmdata/pp/WRFnesting_stg/animation/u_contour_h175'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')


""" U_contour in vertical plane """
### U_contour from WRF
vMin, vMax, vDelta = (-8.2, -1, 1.2)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
fig, axs = plt.subplots(figsize=(5.5,5.0), constrained_layout=False)
x_ = np.copy(xSeq[:-1])
z_ = np.copy(zSeq)
v_ = np.copy(init_atmosphere_u[:,128,:]) # 1,9,17
# v_ -= v_.mean()
# v_[np.where(v_ < vMin)] = vMin
# v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(x_, z_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{U}$" + ' (m/s)', fontsize=12)
plt.xlim([0,2560])
plt.ylim([0,800])
plt.ylabel('z (m)', fontsize=12)
plt.xlabel('x (m)', fontsize=12)
plt.title('')
saveName = 'U_contour_v.png'
saveDir = '/scratch/palmdata/pp/WRFnesting/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close('all')


jobName  = 'WRFnesting_stg'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_0, xSeq_0, ySeq_0, zSeq_0 = getInfo_palm(jobDir, jobName, 'M02', '.001', 'u')
t0 = tSeq_0[0]

tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0 = getData_palm(jobDir, jobName, 'M02', ['.001'], 'u', [0,60], (0,xSeq_0.size), (0,1), (1,zSeq_0.size))
print(np.min(varSeq_0-varSeq_0.mean()),np.max(varSeq_0-varSeq_0.mean())) # find min and max


vMin, vMax, vDelta = (-1.0, 1.0, 0.2)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tInd in range(0,60,1):
    fig, axs = plt.subplots(figsize=(5.5,5.0), constrained_layout=False)
    x_ = xSeq_0
    z_ = zSeq_0
    v_ = np.copy(varSeq_0[tInd,:,0,:])
    v_mean = np.average(init_atmosphere_u.reshape(96,256*255),axis=1)[:81]
    for i in range(zSeq_0.size):
        v_[i] -= v_mean[i]
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, z_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq_0[tInd]-t0,2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{U}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,2560])
    plt.ylim([0,800])
    plt.ylabel('z (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/palmdata/pp/WRFnesting_stg/animation/u_contour_v'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')
