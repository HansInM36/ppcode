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


def getData_sowfa(dir, prbg, trs_para, var, varD):
    """ extract velocity data of specified probe groups """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]

    # read data
    readDir = dir + '/data/'
    readName = prbg
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    coors = data_org['coors']

    # coordinate transformation
    prbNum = coors.shape[0]
    for p in range(prbNum):
        tmp = data_org[var][p]
        data_org[var][p] = funcs.trs(tmp,O,alpha)

    xSeq = np.array(data_org['info'][2])
    ySeq = np.array(data_org['info'][3])
    zSeq = np.array(data_org['info'][4])
    xNum = xSeq.size
    yNum = ySeq.size
    zNum = zSeq.size
    varSeq = data_org[var][:,:,varD]
    tSeq = data_org['time']
    tNum = tSeq.size
    return tSeq, xSeq, ySeq, zSeq, varSeq, coors
def PSD_sowfa(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    data = varSeq
    tNum = tSeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    mean_ = np.mean(varSeq[pInd_start:pInd_end])
    var_ = np.var(varSeq[pInd_start:pInd_end])

    # coors = data_org['coors'][xNum*yNum*zInd:xNum*yNum*(zInd+1)]
    # pNum = coors.shape[0]

    PSD_list = []
    for p in range(pInd_start, pInd_end):
        vSeq = varSeq[p]
        # interpolate
        f = interp1d(tSeq, vSeq, kind='linear', fill_value='extrapolate')
        v_seq = f(t_seq)
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(t_seq, v_seq, deg=deg_))
        tmp = v_seq - polyFunc(t_seq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = funcs.window_weight(tmp)
        # FFT
        # omega_seq, tmp = PSD_omega(t_seq,tmp)
        f_seq, tmp = scipy.signal.csd(tmp, tmp, fs, nperseg=segNum, noverlap=None)
        PSD_list.append(tmp)
    PSD_seq = np.average(np.array(PSD_list), axis=0)

    return f_seq, PSD_seq, mean_, var_

def getData_palm(dir, jobName, maskID, run_no_list, var):
    """ extract velocity data of specified probe groups """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
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
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq
def PSD_palm(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):

    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    # fs = 10
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    mean_ = np.mean(varSeq[:,zInd,:,:])
    var_ = np.var(varSeq[:,zInd,:,:])

    PSD_list = []
    for yInd in range(yNum):
        for xInd in range(xNum):
            vSeq = varSeq[:,zInd,yInd,xInd]
            # interpolate
            f = interp1d(tSeq, vSeq, kind='linear', fill_value='extrapolate')
            v_seq = f(t_seq)
            # detrend
            deg_ = 1
            polyFunc = np.poly1d(np.polyfit(t_seq, v_seq, deg=deg_))
            tmp = v_seq - polyFunc(t_seq)
            tmp = tmp - tmp.mean()
            # bell tapering
            tmp = funcs.window_weight(tmp)
            # FFT
            # omega_seq, tmp = PSD_omega(t_seq,tmp)
            f_seq, tmp = scipy.signal.csd(tmp, tmp, fs, nperseg=segNum, noverlap=None)
            PSD_list.append(tmp)
    PSD_seq = np.average(np.array(PSD_list), axis=0)

    return f_seq, PSD_seq, mean_, var_

def corr_x_av_sowfa(vSeq, tSeq, zInd, ySeq, xSeq, dInd):
    dt = tSeq[-1] - tSeq[-2]
    xNum = xSeq.size
    yNum = ySeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)

    corr = []
    for p in range(pInd_end - pInd_start - dInd):
        u0 = vSeq[p]
        u1 = vSeq[p+dInd]
        tmp = funcs.crosscorr_FFT(u1,u0,dt)
        corr.append(tmp[1])
    tau = tmp[0]
    corr = np.average(np.array(corr),axis=0)
    return tau, corr
def corr_x_av_palm(vSeq, tSeq, zInd, yInd, xSeq, dInd):
    dt = tSeq[-1] - tSeq[-2]

    corr = []
    for p in range(xSeq.size - dInd):
        u0 = vSeq[:,zInd,yInd,p]
        u1 = vSeq[:,zInd,yInd,p+dInd]
        tmp = funcs.crosscorr_FFT(u1,u0,dt)
        corr.append(tmp[1])
    tau = tmp[0]
    corr = np.average(np.array(corr),axis=0)
    return tau, corr

segNum = 20800

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_0, PSD_u_seq_0, u_mean_0, u_var_0 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
# f_seq_0, PSD_u_seq_0 = PSD_sowfa((144000.0, 145200, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, 10240)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs10'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.022'], 'u')
f_seq_1, PSD_u_seq_1, u_mean_1, u_var_1 = PSD_palm((288000.0, 290400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
# f_seq_1, PSD_u_seq_1 = PSD_palm((288000.0, 289200, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, 10240)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs20'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_2, PSD_u_seq_2, u_mean_2, u_var_2 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 2, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs20'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3 = getData_palm(dir_3, jobName, 'M03', ['.002','.003','.004'], 'u')
f_seq_3, PSD_u_seq_3, u_mean_3, u_var_3 = PSD_palm((432000.0, 434400, 0.1), tSeq_3, 3, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_4 = 'gs40'
ppDir_4 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_4
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4, coors_4 = getData_sowfa(ppDir_4, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_4, PSD_u_seq_4, u_mean_4, u_var_4 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_4, 4, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs40'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = getData_palm(dir_5, jobName, 'M03', ['.001'], 'u')
f_seq_5, PSD_u_seq_5, u_mean_5, u_var_5 = PSD_palm((144000.0, 146400, 0.1), tSeq_5, 1, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_6 = 'gs10_refined'
ppDir_6 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_6
tSeq_6, xSeq_6, ySeq_6, zSeq_6, uSeq_6, coors_6 = getData_sowfa(ppDir_6, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq_6, PSD_u_seq_6, u_mean_6, u_var_6 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_6, 4, xSeq_6.size, ySeq_6.size, uSeq_6, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_mdf'
dir_7 = prjDir + '/' + jobName
tSeq_7, xSeq_7, ySeq_7, zSeq_7, uSeq_7 = getData_palm(dir_7, jobName, 'M03', ['.001','.002'], 'u')
f_seq_7, PSD_u_seq_7, u_mean_7, u_var_7 = PSD_palm((72000.0, 74400, 0.1), tSeq_7, 4, xSeq_7.size, ySeq_7.size, uSeq_7, segNum)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL_main'
dir_8 = prjDir + '/' + jobName
tSeq_8, xSeq_8, ySeq_8, zSeq_8, uSeq_8 = getData_palm(dir_8, jobName, 'M03', ['.006','.007'], 'u')
f_seq_8, PSD_u_seq_8, u_mean_8, u_var_8 = PSD_palm((75890.0, 77800, 0.1), tSeq_8, 4, xSeq_8.size, ySeq_8.size, uSeq_8, 15000)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL'
dir_9 = prjDir + '/' + jobName
tSeq_9, xSeq_9, ySeq_9, zSeq_9, uSeq_9 = getData_palm(dir_9, jobName, 'M03', ['.001','.002'], 'u')
f_seq_9, PSD_u_seq_9, u_mean_9, u_var_9 = PSD_palm((144000.0, 146400, 0.1), tSeq_9, 4, xSeq_9.size, ySeq_9.size, uSeq_9, 15000)







### compute advection velocity
tau_6, corr_6 = corr_x_av_sowfa(uSeq_6, tSeq_6, 4, ySeq_6, xSeq_6, 2)
tau_0, corr_0 = corr_x_av_sowfa(uSeq_0, tSeq_0, 4, ySeq_0, xSeq_0, 2)
tau_2, corr_2 = corr_x_av_sowfa(uSeq_2, tSeq_2, 4, ySeq_2, xSeq_2, 2)
ua_6 = 40/tau_6[np.where(corr_6 == np.max(corr_6))[0][0]]
ua_0 = 40/tau_0[np.where(corr_0 == np.max(corr_0))[0][0]]
ua_2 = 40/tau_2[np.where(corr_2 == np.max(corr_2))[0][0]]

tau_7, corr_7 = corr_x_av_palm(uSeq_7, tSeq_7, 4, 0, xSeq_7, 2)
tau_1, corr_1 = corr_x_av_palm(uSeq_1, tSeq_1, 4, 0, xSeq_1, 2)
tau_3, corr_3 = corr_x_av_palm(uSeq_3, tSeq_3, 4, 0, xSeq_3, 2)
ua_7 = 40/tau_7[np.where(corr_7 == np.max(corr_7))[0][0]]
ua_1 = 40/tau_1[np.where(corr_1 == np.max(corr_1))[0][0]]
ua_3 = 40/tau_3[np.where(corr_3 == np.max(corr_3))[0][0]]


# plot
fig, ax = plt.subplots(figsize=(6,6))
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

### original
dfInd = 1
# plt.loglog(f_seq_6[::dfInd], PSD_u_seq_6[::dfInd], label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_0[::dfInd], PSD_u_seq_0[::dfInd], label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
# plt.loglog(f_seq_2[::dfInd], PSD_u_seq_2[::dfInd], label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
# plt.loglog(f_seq_4[::dfInd], PSD_u_seq_4[::dfInd], label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
# plt.loglog(f_seq_7[::dfInd], PSD_u_seq_7[::dfInd], label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
# plt.loglog(f_seq_1[::dfInd], PSD_u_seq_1[::dfInd], label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
# plt.loglog(f_seq_3[::dfInd], PSD_u_seq_3[::dfInd], label='palm-gs20', linewidth=1.0, linestyle='-', color=colors_palm[2])
# plt.loglog(f_seq_5[::dfInd], PSD_u_seq_5[::dfInd], label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])
plt.loglog(f_seq_8[::dfInd], PSD_u_seq_8[::dfInd], label='palm-main-gs10', linewidth=1.0, linestyle='-', color=colors_palm[3])
plt.loglog(f_seq_9[::dfInd], PSD_u_seq_9[::dfInd], label='palm-pcr-gs10', linewidth=1.0, linestyle='-', color=colors_palm[4])

# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')', fontsize=12)

# ### scaled
# alpha =  0.46
# dfInd = 1
# plt.loglog(f_seq_6[::dfInd]*zSeq_6[4]/u_mean_6, PSD_u_seq_6[::dfInd]*f_seq_6[::dfInd]/u_var_6, label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
# plt.loglog(f_seq_0[::dfInd]*zSeq_0[4]/u_mean_0, PSD_u_seq_0[::dfInd]*f_seq_0[::dfInd]/u_var_0, label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
# plt.loglog(f_seq_2[::dfInd]*zSeq_2[2]/u_mean_2, PSD_u_seq_2[::dfInd]*f_seq_2[::dfInd]/u_var_2, label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
# # plt.loglog(f_seq_4[::dfInd]*zSeq_4[4]/u_mean_4, PSD_u_seq_4[::dfInd]*f_seq_4[::dfInd]/u_var_4, label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
# plt.loglog(f_seq_7[::dfInd]*zSeq_7[4]/u_mean_7/alpha, PSD_u_seq_7[::dfInd]*f_seq_7[::dfInd]/u_var_7, label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
# plt.loglog(f_seq_1[::dfInd]*zSeq_1[4]/u_mean_1/alpha, PSD_u_seq_1[::dfInd]*f_seq_1[::dfInd]/u_var_1, label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
# plt.loglog(f_seq_3[::dfInd]*zSeq_3[3]/u_mean_3/alpha, PSD_u_seq_3[::dfInd]*f_seq_3[::dfInd]/u_var_3, label='palm-gs20', linewidth=1.0, linestyle='-', color=colors_palm[2])
# # plt.loglog(f_seq_5[::dfInd]*zSeq_5[1]/u_mean_5/alpha, PSD_u_seq_5[::dfInd]*f_seq_5[::dfInd]/u_var_5, label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])
#
# # plt.loglog(f_seq_6[::dfInd]*zSeq_6[4]/ua_6, PSD_u_seq_6[::dfInd]*f_seq_6[::dfInd]/u_var_6, label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
# # plt.loglog(f_seq_0[::dfInd]*zSeq_0[4]/ua_0, PSD_u_seq_0[::dfInd]*f_seq_0[::dfInd]/u_var_0, label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
# # plt.loglog(f_seq_2[::dfInd]*zSeq_2[2]/ua_2, PSD_u_seq_2[::dfInd]*f_seq_2[::dfInd]/u_var_2, label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
# # # plt.loglog(f_seq_4[::dfInd]*zSeq_4[4]/ua_4, PSD_u_seq_4[::dfInd]*f_seq_4[::dfInd]/u_var_4, label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
# # plt.loglog(f_seq_7[::dfInd]*zSeq_7[4]/ua_7, PSD_u_seq_7[::dfInd]*f_seq_7[::dfInd]/u_var_7, label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
# # plt.loglog(f_seq_1[::dfInd]*zSeq_1[4]/ua_1, PSD_u_seq_1[::dfInd]*f_seq_1[::dfInd]/u_var_1, label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
# # plt.loglog(f_seq_3[::dfInd]*zSeq_3[3]/ua_3, PSD_u_seq_3[::dfInd]*f_seq_3[::dfInd]/u_var_3, label='palm-gs20', linewidth=1.0, linestyle='-', color=colors_palm[2])
# # # plt.loglog(f_seq_5[::dfInd]*zSeq_5[1]/ua_5/alpha, PSD_u_seq_5[::dfInd]*f_seq_5[::dfInd]/u_var_5, label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])

# # -5/3 law
# f_ = np.linspace(1e-1,1e1,100)
# plt.loglog(f_, 1e0*np.power(f_, -2/3), label='-2/3 law', linewidth=2.0, color='k')
# plt.xlabel(r'$\mathrm{fz/\overline{u}}$', fontsize=12)
# plt.ylabel(r"$\mathrm{fS_u/\sigma^2_u}$", fontsize=12)
# xaxis_min = 1e-2
# xaxis_max = 1e2 # f_seq.max()
# yaxis_min = 1e-16
# yaxis_max = 1e2
# plt.ylim(yaxis_min, yaxis_max)
# plt.xlim(xaxis_min, xaxis_max)

# print(u_mean_6, u_mean_0, u_mean_2, u_mean_4, u_mean_7, u_mean_1, u_mean_3, u_mean_5)
# print(u_var_6, u_var_0, u_var_2, u_var_4, u_var_7, u_var_1, u_var_3, u_var_5)


# plt.legend(bbox_to_anchor=(0.02,0.42), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(0.05,0.26), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()








tau_6_0, corr_6_0 = corr_x_av_sowfa(uSeq_6, tSeq_6, 4, ySeq_6, xSeq_6, 2)
tau_6_1, corr_6_1 = corr_x_av_sowfa(uSeq_6, tSeq_6, 4, ySeq_6, xSeq_6, 4)

tau_7_0, corr_7_0 = corr_x_av_palm(uSeq_7, tSeq_7, 4, 0, xSeq_7, 2)
tau_7_1, corr_7_1 = corr_x_av_palm(uSeq_7, tSeq_7, 4, 0, xSeq_7, 4)

fig, ax = plt.subplots(figsize=(6,6))
plt.plot(tau_6_0, corr_6_0, 'r-', label='sowfa-40m')
plt.plot(tau_6_1, corr_6_1, 'r--', label='sowfa-80m')
plt.plot(tau_7_0, corr_7_0, 'b-', label='palm-40m')
plt.plot(tau_7_1, corr_7_1, 'b--', label='palm-80m')
plt.vlines(6.2, -0.2, 1, linestyle='-', linewidth=1, color='r')
plt.vlines(12.3, -0.2, 1, linestyle='--', linewidth=1, color='r')
plt.vlines(9.6, -0.2, 1, linestyle='-', linewidth=1, color='b')
plt.vlines(18.6, -0.2, 1, linestyle='--', linewidth=1, color='b')
xaxis_min = 0.0
xaxis_max = 100.0 # f_seq.max()
yaxis_min = -0.2
yaxis_max = 1.0
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.xlabel(r'$\mathrm{\tau}$ (s)', fontsize=12)
plt.ylabel('coor', fontsize=12)
plt.grid()
plt.legend(bbox_to_anchor=(0.66,0.86), loc=6, borderaxespad=0)
fig.tight_layout()
plt.show()
