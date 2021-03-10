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
from matplotlib import colors


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



jobName = 'WRFPALM_20150701'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq, zSeq, uSeq = velo_pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'u')
tSeq, zSeq, vSeq = velo_pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'v')

uvSeq = np.sqrt(np.power(uSeq,2) + np.power(vSeq,2))
wdSeq = 270 - np.arctan(vSeq / (uSeq + 1e-6)) * 180/np.pi

#### checking
#single_plot(uSeq_0[-1], zSeq_0)
#u_90 = ITP(uSeq_0[-1], zSeq_0, 90)
#v_90 = ITP(vSeq_0[-1], zSeq_0, 90)


""" profile of horizontal velocity """
fig, ax = plt.subplots(figsize=(6,4.5))
colors = plt.cm.jet(np.linspace(0,1,tSeq.size))
for i in range(tSeq.size):
    plt.plot(uvSeq[i], zSeq, label='t = ' + str(int(tSeq[i])) + 's', linewidth=1.0, color=colors[i])
plt.xlabel(r"$\mathrm{\overline{u}_h}$ (m/s)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 0.0
xaxis_max = 16.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1200.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'velo_pr.png'
saveDir = '/scratch/projects/WRFnesting/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" wind direction profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(6,4.5))
colors = plt.cm.jet(np.linspace(0,1,tSeq.size))

for i in range(tSeq.size):
    plt.plot(wdSeq[i,1:], zSeq[1:], label='t = ' + str(int(tSeq[i])) + 's', linewidth=1.0, color=colors[i])

plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 260.0
xaxis_max = 340.0
xaxis_d = 10
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'wd_pr.png'
saveDir = '/scratch/projects/WRFnesting/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()



#""" dimensionless u gradient profile of stationary flow """
#startH = 5.000
#topH = 205.0
#zNum_ = 21
#kappa = 0.4
#uStar_0 = kappa / np.log(zSeq_0[0]/0.001) * np.power(uSeq_0[-1][0]**2 + vSeq_0[-1][0]**2,0.5)
#
#fig, ax = plt.subplots(figsize=(3,4.5))
#z_ = np.linspace(startH,topH,zNum_)
#dz = (topH - startH) / (zNum_-1)
## sowfa
#zero = np.zeros(1)
#v_0 = np.concatenate((zero, uSeq_0[-1]))
#z_0 = np.concatenate((zero, zSeq_0))
#f_0 = interp1d(z_0, v_0, kind='linear', fill_value='extrapolate')
#v_0 = funcs.calc_deriv_1st_FD(dz, f_0(z_))
#v_0 = v_0 * kappa * z_ / uStar_0
## palm
#v_3 = uSeq_3[-1]
#z_3 = zSeq_3
#f_3 = interp1d(z_3, v_3, kind='linear', fill_value='extrapolate')
#v_3 = funcs.calc_deriv_1st_FD(dz, f_3(z_))
#v_3 = v_3 * kappa * z_ / uStar_3
#
#plt.plot(v_0, z_, label='sowfa', linewidth=1.0, linestyle='-', color='k')
#plt.plot(v_3, z_, label='palm', linewidth=1.0, linestyle='--', color='k')
#plt.xlabel(r"$\mathrm{\phi_m}$", fontsize=12)
#plt.ylabel('z (m)', fontsize=12)
#xaxis_min = -3
#xaxis_max = 5
#xaxis_d = 2
#yaxis_min = 0
#yaxis_max = 200.0
#yaxis_d = 20.0
#plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
#plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
#plt.grid()
#plt.title('')
#fig.tight_layout() # adjust the layout
## saveName = 'phi_m' + '_pr.png'
## plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
#plt.show()
#plt.close()
