#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
import os
import sys
import pickle
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt
from matplotlib import colors

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=int(days)) \
           + datetime.timedelta(hours=int(hours)) \
           + datetime.timedelta(minutes=int(minutes)) \
           + datetime.timedelta(seconds=round(seconds)) \
           - datetime.timedelta(days=366)

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


""" ppWRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRFpp'
readName = "WRFOUT_NODA_20150701.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimList = list(data.dimensions)
varList = list(data.variables)

time_org = data.variables['time'][:]
dateTime = []
time = []
for tInd in range(time_org.size):
    dateTime.append(datenum_to_datetime(time_org[tInd]))
    time.append(dateTime[tInd].timestamp() - dateTime[0].timestamp())

Z = data.variables['Z'][:]
U = data.variables['U'][:]
V = data.variables['V'][:]
THETA = data.variables['THETA'][:]
Q = data.variables['Q'][:]

UV = np.sqrt(np.power(U,2) + np.power(V,2))
WD = funcs.wd(U,V)



#jobName = 'EERASP3_ndg_1'
#dir = '/scratch/palmdata/JOBS/' + jobName
#u = pr_palm(dir, jobName, ['.000','.001','.002','.003'], 'u')
#v = pr_palm(dir, jobName, ['.000','.001','.002','.003'], 'v')
#theta = pr_palm(dir, jobName, ['.000','.001','.002','.003'], 'theta')

jobName = 'EERASP3_2_ndg'
dir = '/scratch/palmdata/JOBS/' + jobName
u = pr_palm(dir, jobName, ['.000','.001'], 'u')
v = pr_palm(dir, jobName, ['.000','.001'], 'v')
theta = pr_palm(dir, jobName, ['.000','.001'], 'theta')

uv = np.sqrt(np.power(u[2],2) + np.power(v[2],2))
wd = funcs.wd(u[2], v[2])

""" profile of u - cmp """
for tInd in range(u[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    plt.plot(u[2][tInd], u[1], label='ndg', linestyle='-', linewidth=1.0, color='k')
    plt.plot(U[132], Z[132], label='WRF', marker='', linestyle='--', linewidth=1.0, color='k')
    plt.xlabel(r"$\mathrm{\overline{u}}$ (m/s)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
#    xaxis_min = -10.0
#    xaxis_max = 0.0
#    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.6,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'pr_cmp_t' + str(np.round(u_0[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/pr/cmp'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()

""" profile of v - cmp """
for tInd in range(v[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    plt.plot(v[2][tInd], v[1], label='ndg', linestyle='-', linewidth=1.0, color='k')
    plt.plot(V[132], Z[132], label='WRF', marker='', linestyle='--', linewidth=1.0, color='k')
    plt.xlabel(r"$\mathrm{\overline{v}}$ (m/s)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
#    xaxis_min = -2.0
#    xaxis_max = 12.0
#    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.6,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(v[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'pr_cmp_t' + str(np.round(u_0[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/pr/cmp'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()

""" profile of horizontal velocity - cmp """
for tInd in range(u[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    plt.plot(uv[tInd], u[1], label='ndg', linestyle='-', linewidth=1.0, color='k')
    plt.plot(UV[132], Z[132], label='WRF', marker='', linestyle='--', linewidth=1.0, color='k')
    plt.xlabel(r"$\mathrm{\overline{u}_h}$ (m/s)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
#    xaxis_min = 0.0
#    xaxis_max = 16.0
#    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.6,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'velo_pr_cmp_wrf_ndg' + str(np.round(u[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/velo_pr_cmp_wrf_ndg'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of horizontal wind direction """
for tInd in range(u[0].size):
    fig, ax = plt.subplots(figsize=(3,4.5))
    print('PALM time: ' + str(u[0][tInd]) + ', ' + 'WRF time: ' + str(time[tInd]))
    plt.plot(wd[tInd][1:], u[1][1:], label='PALM', linewidth=1.0, color='k')
    plt.plot(WD[132][1:], Z[132][1:], label='WRF', marker='', linestyle='--', linewidth=1.0, color='k')
    plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
#    xaxis_min = 80.0
#    xaxis_max = 200.0
#    xaxis_d = 30
    yaxis_min = 0
    yaxis_max = 1000.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
#    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
#    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'cmp_wd_pr_' + str(u[0][tInd]//3600) + 'hr' + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_wd_pr'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of theta - cmp """
for tInd in range(theta[0].size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    plt.plot(theta[2][tInd], theta[1], label='ref', linestyle='-', linewidth=1.0, color='k')
    plt.xlabel(r"$\mathrm{\overline{\theta}}$ (K)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 20.0
    xaxis_max = 320.0
    xaxis_d = 100.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.6,0.76), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(theta[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'pr_cmp_t' + str(np.round(u_0[0][tInd]/3600,1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/pr/cmp'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()