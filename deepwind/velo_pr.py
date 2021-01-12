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
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, zSeq, varSeq
def velo_pr_sowfa(dir, trs_para, varD):
    """ extract horizontal average of velocity at various times and heights """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]

    fr = open(dir + '/data/' + 'aveData', 'rb')
    aveData = pickle.load(fr)
    fr.close()

    zSeq = aveData['H']
    zNum = zSeq.size

    tSeq = aveData['time']
    tNum = tSeq.size
    tDelta = tSeq[1] - tSeq[0]

    uSeq = aveData['U_mean']
    vSeq = aveData['V_mean']
    wSeq = aveData['W_mean']

    varSeq = np.zeros((tNum,zNum))
    for zInd in range(zNum):
        tmp = np.concatenate((uSeq[:,zInd].reshape(tNum,1), vSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp = np.concatenate((tmp, wSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp_ = funcs.trs(tmp,O,alpha)
        varSeq[:,zInd] = tmp_[:,varD]

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


prjName = 'deepwind'
jobName_0 = 'gs10'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, uSeq_0 =  velo_pr_sowfa(dir_0, ((0,0,0),30.0), 0)
tSeq_0, zSeq_0, vSeq_0 =  velo_pr_sowfa(dir_0, ((0,0,0),30.0), 1)
t_seq_0, uSeq_0 = velo_pr_ave((3600,144000,144000,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, uSeq_0)
t_seq_0, vSeq_0 = velo_pr_ave((3600,144000,144000,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, vSeq_0)

prjName = 'deepwind'
jobName_0 = 'gs10_0.0001'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, uSeq_0 =  velo_pr_sowfa(dir_0, ((0,0,0),30.0), 0)
tSeq_0, zSeq_0, vSeq_0 =  velo_pr_sowfa(dir_0, ((0,0,0),30.0), 1)
t_seq_0, uSeq_0 = velo_pr_ave((2400,144000,146400,2400), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, uSeq_0)
t_seq_0, vSeq_0 = velo_pr_ave((2400,144000,146400,2400), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, vSeq_0)


prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'deepwind_NBL'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, zSeq_1, uSeq_1 = velo_pr_palm(dir_1, jobName_1, ['.000', '.001'], 'u')
tSeq_1, zSeq_1, vSeq_1 = velo_pr_palm(dir_1, jobName_1, ['.000', '.001'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_2  = 'deepwind_gs10_mdf'
dir_2 = prjDir + '/' + jobName_2
tSeq_2, zSeq_2, uSeq_2 = velo_pr_palm(dir_2, jobName_2, ['.000', '.001'], 'u')
tSeq_2, zSeq_2, vSeq_2 = velo_pr_palm(dir_2, jobName_2, ['.000', '.001'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_3  = 'deepwind_gs5'
dir_3 = prjDir + '/' + jobName_3
tSeq_3, zSeq_3, uSeq_3 = velo_pr_palm(dir_3, jobName_3, ['.000', '.001','.002','.003','.004','.005','.006','.007','.008','.009','.010','.011'], 'u')
tSeq_3, zSeq_3, vSeq_3 = velo_pr_palm(dir_3, jobName_3, ['.000', '.001','.002','.003','.004','.005','.006','.007','.008','.009','.010','.011'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_4  = 'deepwind_NBL_main'
dir_4 = prjDir + '/' + jobName_4
tSeq_4, zSeq_4, uSeq_4 = velo_pr_palm(dir_4, jobName_4, ['.000', '.001','.002','.003','.004'], 'u')
tSeq_4, zSeq_4, vSeq_4 = velo_pr_palm(dir_4, jobName_4, ['.000', '.001','.002','.003','.004'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_5  = 'deepwind_gs10_0.0001'
dir_5 = prjDir + '/' + jobName_5
tSeq_5, zSeq_5, uSeq_5 = velo_pr_palm(dir_5, jobName_5, ['.000'], 'u')
tSeq_5, zSeq_5, vSeq_5 = velo_pr_palm(dir_5, jobName_5, ['.000'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_6  = 'deepwind_gs20_0.01'
dir_6 = prjDir + '/' + jobName_6
tSeq_6, zSeq_6, uSeq_6 = velo_pr_palm(dir_6, jobName_6, ['.000'], 'u')
tSeq_6, zSeq_6, vSeq_6 = velo_pr_palm(dir_6, jobName_6, ['.000'], 'v')

prjDir = '/scratch/palmdata/JOBS'
jobName_7  = 'deepwind_gs5_mdf'
dir_7 = prjDir + '/' + jobName_7
tSeq_7, zSeq_7, uSeq_7 = velo_pr_palm(dir_7, jobName_7, ['.000','.001'], 'u')
tSeq_7, zSeq_7, vSeq_7 = velo_pr_palm(dir_7, jobName_7, ['.000','.001'], 'v')

### checking
single_plot(uSeq_0[-1], zSeq_0)
u_90 = ITP(uSeq_0[-1], zSeq_0, 90)
v_90 = ITP(vSeq_0[-1], zSeq_0, 90)


""" u profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(3,4.5))

plt.plot(uSeq_0[-1], zSeq_0, label='sowfa', linewidth=1.0, linestyle='-', color='k')
# plt.plot(uSeq_1[-1], zSeq_1, label='palm', linewidth=1.0, linestyle='-', color='r')
# plt.plot(uSeq_3[-1], zSeq_3, label='palm', linewidth=1.0, linestyle='-', color='b')
# plt.plot(uSeq_5[-1], zSeq_5, label='palm', linewidth=1.0, linestyle='-', color='g')
# plt.plot(uSeq_6[-1], zSeq_6, label='palm', linewidth=1.0, linestyle='-', color='k')
# plt.axhline(y=hubH, ls='--', c='black')
# plt.axhline(y=dampH, ls=':', c='black')
plt.xlabel(r"$\overline{\mathrm{u}}$ (m/s)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 0
xaxis_max = 12.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'u' + '_pr.png'
# plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()


""" dimensionless u gradient profile of stationary flow (sowfa vs palm) """
startH = 5.000
topH = 205.0
zNum_ = 21
kappa = 0.4
uStar_0 = kappa / np.log(zSeq_0[0]/0.001) * np.power(uSeq_0[-1][0]**2 + vSeq_0[-1][0]**2,0.5)
uStar_1 = kappa / np.log(zSeq_1[1]/0.001) * np.power(uSeq_1[-1][1]**2 + vSeq_1[-1][1]**2,0.5)

fig, ax = plt.subplots(figsize=(3,4.5))
z_ = np.linspace(startH,topH,zNum_)
dz = (topH - startH) / (zNum_-1)
# sowfa
zero = np.zeros(1)
v_0 = np.concatenate((zero, uSeq_0[-1]))
z_0 = np.concatenate((zero, zSeq_0))
f_0 = interp1d(z_0, v_0, kind='linear', fill_value='extrapolate')
v_0 = funcs.calc_deriv_1st_FD(dz, f_0(z_))
v_0 = v_0 * kappa * z_ / uStar_0
# palm
v_1 = uSeq_1[-1]
z_1 = zSeq_1
f_1 = interp1d(z_1, v_1, kind='linear', fill_value='extrapolate')
v_1 = funcs.calc_deriv_1st_FD(dz, f_1(z_))
v_1 = v_1 * kappa * z_ / uStar_1

plt.plot(v_0, z_, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.plot(v_1, z_, label='palm', linewidth=1.0, linestyle='--', color='k')
plt.xlabel(r"$\mathrm{\phi_m}$", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = -3
xaxis_max = 5
xaxis_d = 2
yaxis_min = 0
yaxis_max = 200.0
yaxis_d = 20.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'phi_m' + '_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()


""" wind direction profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsi""" animation for deepwind_gs10 (SOWFA) """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq, xSeq, ySeq, H, varSeq = getSliceData_Nz_sowfa(ppDir, jobName, 'Nz0', 'U', 0, ((0,0,0),30), (0,150), (0,2560,256), (0,2560,256))
t0 = tSeq[0]

vMin, vMax, vDelta = (-2, 2, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

# transform coors of prbg
prbg0 = np.vstack((780 + 20*np.arange(0,51), np.array([1280 for i in range(51)]), np.array([0 for i in range(51)])))
prbg0 = funcs.trs(prbg0.T, (1280,1280,0), -30); prbg0[:,0] += 1280; prbg0[:,1] += 1280;
prbg1 = np.vstack((np.array([1280 for i in range(51)]), 780 + 20*np.arange(0,51), np.array([0 for i in range(51)])))
prbg1 = funcs.trs(prbg1.T, (1280,1280,0), -30); prbg1[:,0] += 1280; prbg1[:,1] += 1280;


for tInd in range(0,150,100):
    fig, axs = plt.subplots(figsize=(8,8), constrained_layout=False)
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
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/projects/deepwind/photo/animation/Nz_sowfa'
    # if not os.path.exists(saveDir):
    #     os.makedirs(saveDir)
    # plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()ze=(3,4.5))

wdSeq_0 = 270 - np.arctan(vSeq_0[-1] / uSeq_0[-1]) * 180/np.pi
wdSeq_1 = 270 - np.arctan(vSeq_1[-1][1:] / uSeq_1[-1][1:]) * 180/np.pi

plt.plot(wdSeq_0, zSeq_0, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.plot(wdSeq_1, zSeq_1[1:], label='palm', linewidth=1.0, linestyle='--', color='k')
# plt.axhline(y=hubH, ls='--', c='black')
# plt.axhline(y=dampH, ls=':', c='black')
plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
# plt.ylabel('z (m)')
xaxis_min = 260
xaxis_max = 290
xaxis_d = 5
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), [], fontsize=12)
plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'wd' + '_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()
