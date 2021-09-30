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






prjDir = '/scratch/palmdata/JOBS'
jobName  = 'example_nbl'
jobDir = prjDir + '/' + jobName
tSeq, zSeq, uSeq = velo_pr_palm(jobDir, jobName, ['.000'], 'u')
tSeq, zSeq, vSeq = velo_pr_palm(jobDir, jobName, ['.000'], 'v')
tSeq, zwSeq, wSeq = velo_pr_palm(jobDir, jobName, ['.000'], 'w')


""" u profile of stationary flow (last time step) """
fig, ax = plt.subplots(figsize=(3,4.5))

plt.plot(uSeq[-1], zSeq, label='palm', linewidth=1.0, linestyle='-', color='k')
plt.xlabel(r"$\overline{\mathrm{u}}$ (m/s)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
#xaxis_min = 0
#xaxis_max = 12.0
#xaxis_d = 2.0
#yaxis_min = 0.0
#yaxis_max = 1.0
#yaxis_d = 0.1
#plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'u' + '_pr.png'
# saveDir = '/scratch/projects/deepwind/photo/profiles'
# plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()



""" wind direction profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(3,4.5))

wdSeq_0 = 270 - np.arctan(vSeq_0[-1] / uSeq_0[-1]) * 180/np.pi
wdSeq_3 = 270 - np.arctan(vSeq_3[-1][1:] / uSeq_3[-1][1:]) * 180/np.pi

plt.plot(wdSeq_0, zSeq_0, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.plot(wdSeq_3, zSeq_3[1:], label='palm', linewidth=1.0, linestyle='--', color='k')
# plt.axhline(y=hubH, ls='--', c='black')
# plt.axhline(y=dampH, ls=':', c='black')
plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 260
xaxis_max = 290
xaxis_d = 10
yaxis_min = 0
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
# saveName = 'wd' + '_pr.png'
# plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()

""" print wind speed variation """
wsv_0 = ITP(uSeq_0[-1], zSeq_0, 180) - ITP(uSeq_0[-1], zSeq_0, 20)
wsv_1 = ITP(uSeq_1[-1], zSeq_1, 180) - ITP(uSeq_1[-1], zSeq_1, 20)
wsv_2 = ITP(uSeq_2[-1], zSeq_2, 180) - ITP(uSeq_2[-1], zSeq_2, 20)

wsv_3 = ITP(uSeq_3[-1], zSeq_3, 180) - ITP(uSeq_3[-1], zSeq_3, 20)
wsv_4 = ITP(uSeq_4[-1], zSeq_4, 180) - ITP(uSeq_4[-1], zSeq_4, 20)
wsv_5 = ITP(uSeq_5[-1], zSeq_5, 180) - ITP(uSeq_5[-1], zSeq_5, 20)

print('sowfa-0.0001: ', np.round(wsv_0,3))
print('sowfa-0.001: ', np.round(wsv_1,3))
print('sowfa-0.01: ', np.round(wsv_2,3))

print('palm-0.0001: ', np.round(wsv_3,3))
print('palm-0.001: ', np.round(wsv_4,3))
print('palm-0.01: ', np.round(wsv_5,3))


""" print wind direction variation """
wdSeq_0 = 270 - np.arctan(vSeq_0[-1] / uSeq_0[-1]) * 180/np.pi
wdSeq_1 = 270 - np.arctan(vSeq_1[-1] / uSeq_1[-1]) * 180/np.pi
wdSeq_2 = 270 - np.arctan(vSeq_2[-1] / uSeq_2[-1]) * 180/np.pi

wdSeq_3 = 270 - np.arctan(vSeq_3[-1][1:] / uSeq_3[-1][1:]) * 180/np.pi
wdSeq_4 = 270 - np.arctan(vSeq_4[-1][1:] / uSeq_4[-1][1:]) * 180/np.pi
wdSeq_5 = 270 - np.arctan(vSeq_5[-1][1:] / uSeq_5[-1][1:]) * 180/np.pi

wdv_0 = ITP(wdSeq_0, zSeq_0, 180) - ITP(wdSeq_0, zSeq_0, 20)
wdv_1 = ITP(wdSeq_1, zSeq_1, 180) - ITP(wdSeq_1, zSeq_1, 20)
wdv_2 = ITP(wdSeq_2, zSeq_2, 180) - ITP(wdSeq_2, zSeq_2, 20)

wdv_3 = ITP(wdSeq_3, zSeq_3[1:], 180) - ITP(wdSeq_3, zSeq_3[1:], 20)
wdv_4 = ITP(wdSeq_4, zSeq_4[1:], 180) - ITP(wdSeq_4, zSeq_4[1:], 20)
wdv_5 = ITP(wdSeq_5, zSeq_5[1:], 180) - ITP(wdSeq_5, zSeq_5[1:], 20)

print('sowfa-0.0001: ', np.round(wdv_0,3))
print('sowfa-0.001: ', np.round(wdv_1,3))
print('sowfa-0.01: ', np.round(wdv_2,3))

print('palm-0.0001: ', np.round(wdv_3,3))
print('palm-0.001: ', np.round(wdv_4,3))
print('palm-0.01: ', np.round(wdv_5,3))


""" print uStar """
kappa = 0.4

uStar_0 = kappa / np.log(zSeq_0[0]/0.0001) * np.power(uSeq_0[-1][0]**2 + vSeq_0[-1][0]**2,0.5)
uStar_1 = kappa / np.log(zSeq_1[0]/0.001) * np.power(uSeq_1[-1][0]**2 + vSeq_1[-1][0]**2,0.5)
uStar_2 = kappa / np.log(zSeq_2[0]/0.01) * np.power(uSeq_2[-1][0]**2 + vSeq_2[-1][0]**2,0.5)

uStar_3 = kappa / np.log(zSeq_3[1]/0.0001) * np.power(uSeq_3[-1][1]**2 + vSeq_3[-1][1]**2,0.5)
uStar_4 = kappa / np.log(zSeq_4[1]/0.001) * np.power(uSeq_4[-1][1]**2 + vSeq_4[-1][1]**2,0.5)
uStar_5 = kappa / np.log(zSeq_5[1]/0.01) * np.power(uSeq_5[-1][1]**2 + vSeq_5[-1][1]**2,0.5)

print('sowfa-0.0001: ', np.round(uStar_0,3))
print('sowfa-0.001: ', np.round(uStar_1,3))
print('sowfa-0.01: ', np.round(uStar_2,3))

print('palm-0.0001: ', np.round(uStar_3,3))
print('palm-0.001: ', np.round(uStar_4,3))
print('palm-0.01: ', np.round(uStar_5,3))
