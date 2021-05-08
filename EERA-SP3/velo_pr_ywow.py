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


def pr_palm(dir, jobName, run_no_list, var):
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

def pr_sowfa(dir, trs_para, varD):
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

def pr_ave(tplot_para, tSeq, tDelta, zNum, varSeq):
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
    """ single pr plot """
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

#jobName = 'EERASP3_2'
#dir = '/scratch/palmdata/JOBS/' + jobName
#u_0 = pr_palm(dir, jobName, ['.000','.001','.002'], 'u')
#v_0 = pr_palm(dir, jobName, ['.000','.001','.002'], 'v')

jobName = 'EERASP3_1_yw'
dir = '/scratch/palmdata/JOBS/' + jobName
u_1 = pr_palm(dir, jobName, ['.000','.001','.002'], 'u')
v_1 = pr_palm(dir, jobName, ['.000','.001','.002'], 'v')

jobName = 'EERASP3_1_ow'
dir = '/scratch/palmdata/JOBS/' + jobName
u_1_ = pr_palm(dir, jobName, ['.000','.001','.002'], 'u')
v_1_ = pr_palm(dir, jobName, ['.000','.001','.002'], 'v')

#uv_0 = np.sqrt(np.power(u_0[2],2) + np.power(v_0[2],2))
uv_1 = np.sqrt(np.power(u_1[2],2) + np.power(v_1[2],2))
uv_1_ = np.sqrt(np.power(u_1_[2],2) + np.power(v_1_[2],2))

#wd_0 = funcs.wd(u_0[2], v_0[2])
wd_1 = funcs.wd(u_1[2], v_1[2])
wd_1_ = funcs.wd(u_1_[2], v_1_[2])


""" profile of horizontal velocity - certain time - cmp """
for tInd in range(u_1[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
#    plt.plot(uv_0[tInd], u_0[1], label='ref', linestyle='-', linewidth=1.0, color='k')
    plt.plot(uv_1[tInd], u_1[1], label='yw', linestyle='-', linewidth=1.0, color='b')
    plt.plot(uv_1_[tInd], u_1_[1], label='ow', linestyle='-', linewidth=1.0, color='r')
    plt.xlabel(r"$\mathrm{\overline{u}_h}$ (m/s)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 6.0
    xaxis_max = 14.0
    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 200.0
    yaxis_d = 20.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.6,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(np.round(tInd,1)) + 'h')
    fig.tight_layout() # adjust the layout
#    saveName = 'velo_pr_1_ywow_t' + str(np.round(u_1[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/velo_pr/1_ywow'
    saveName = 'velo_pr_1_ywow_t' + str(np.round(u_1[0][tInd]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/velo_pr/1_ywow'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
""" wind direction profile of stationary flow - certain time - cmp """
for tInd in range(u_1[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
#    plt.plot(wd_0[tInd][1:], u_0[1][1:], label='ref', linestyle='-', linewidth=1.0, color='k')
    plt.plot(wd_1[tInd][1:], u_1[1][1:], label='yw', linestyle='-', linewidth=1.0, color='b')
    plt.plot(wd_1_[tInd][1:], u_1_[1][1:], label='ow', linestyle='-', linewidth=1.0, color='r')
    plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 80.0 # 80;240
    xaxis_max = 160.0 # 160;260
    xaxis_d = 10
    yaxis_min = 0
    yaxis_max = 200.0
    yaxis_d = 20.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.12,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(np.round(tInd,1)) + 'h')
    fig.tight_layout() # adjust the layout
#    saveName = 'wd_pr_1_ywow_t' + str(np.round(u_1[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/wd_pr/wd_pr/1_ywow'
    saveName = 'wd_pr_1_ywow_t' + str(np.round(u_1[0][tInd]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/wd_pr/wd_pr/1_ywow'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()