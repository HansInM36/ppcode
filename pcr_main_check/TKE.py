import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt

def TKE_sowfa(dir, trs_para, varD):
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

    uuSeq = aveData['uu_mean']
    vvSeq = aveData['vv_mean']
    wwSeq = aveData['ww_mean']
    R11Seq = aveData['R11_mean']
    R22Seq = aveData['R22_mean']
    R33Seq = aveData['R33_mean']

    rsvSeq = 0.5 * (uuSeq + vvSeq + wwSeq)
    sgsSeq = 0.5 * (R11Seq + R22Seq + R33Seq)

    totSeq = rsvSeq + sgsSeq
    return tSeq, zSeq, rsvSeq, sgsSeq, totSeq
def TKE_plot_sowfa(TKESeq, tSeq, zNum, t_para):
    tDelta = tSeq[1] - tSeq[0]
    ave_itv = t_para[0]
    tplot = t_para[1]

    TKESeq_ = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, TKESeq[:,zInd], kind='linear', fill_value='extrapolate')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        TKESeq_[zInd] = f(tplotSeq).mean()
    return TKESeq_

def TKE_palm(dir, run_no_list):
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    rsvSeq_list = []
    sgsSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        rsvSeq_list.append(np.array(nc_file_list[i].variables['e*'][:], dtype=type(nc_file_list[i].variables['e*'])))
        sgsSeq_list.append(np.array(nc_file_list[i].variables['e'][:], dtype=type(nc_file_list[i].variables['e'])))

    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

    height = list(nc_file_list[0].variables['e*'].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
    zSeq = zSeq.astype(float)
    zNum = zSeq.size

    # concatenate arraies of all run_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size

    rsvSeq = np.concatenate([rsvSeq_list[i] for i in range(run_num)], axis=0)
    rsvSeq = rsvSeq.astype(float)
    sgsSeq = np.concatenate([sgsSeq_list[i] for i in range(run_num)], axis=0)
    sgsSeq = sgsSeq.astype(float)

    totSeq = rsvSeq + sgsSeq
    return tSeq, zSeq, rsvSeq, sgsSeq, totSeq
def TKE_plot_palm(TKESeq, tSeq, zNum, tplot):
    TKESeq = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq[1:], TKESeq[1:,zInd], kind='linear', fill_value='extrapolate')
        TKESeq_[zInd] = f(tplot)
    return TKESeq_

def ITP(varSeq, zSeq, z):
    f = interp1d(zSeq, varSeq, kind='linear')
    return f(z)


""" SOWFA """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'

jobName = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_0, zSeq_0, rsvSeq_0, sgsSeq_0, totSeq_0 = TKE_sowfa(ppDir_0, ((0,0,0),30), 0)
rsvSeq_0 = TKE_plot_sowfa(rsvSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))
sgsSeq_0 = TKE_plot_sowfa(sgsSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))
totSeq_0 = TKE_plot_sowfa(totSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))


""" PALM """
prjDir = '/scratch/palmdata/JOBS'

jobName  = 'deepwind_gs5'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq_4, zSeq_4, rsvSeq_4, sgsSeq_4, totSeq_4 = TKE_palm(dir, ['.010','.011'])
rsvSeq_4 = rsvSeq_4[-1]
sgsSeq_4 = sgsSeq_4[-1]
totSeq_4 = totSeq_4[-1]



""" TKE group plot """
fig = plt.figure()
fig.set_figwidth(5)
fig.set_figheight(5)
rNum, cNum = (1,2)
axs = fig.subplots(nrows=rNum, ncols=cNum)

axs[0].plot(rsvSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-rsv', marker='', markersize=1, linestyle='--', linewidth=1.0, color='r')
axs[0].plot(sgsSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-sgs', marker='', markersize=1, linestyle=':', linewidth=1.0, color='r')
axs[0].plot(totSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-tot', marker='', markersize=1, linestyle='-', linewidth=1.0, color='r')
axs[0].plot(rsvSeq_4, zSeq_4/zi, label='palm-rsv', marker='', markersize=1, linestyle='--', linewidth=1.0, color='b')
axs[0].plot(sgsSeq_4, zSeq_4/zi, label='palm-sgs', marker='', markersize=1, linestyle=':', linewidth=1.0, color='b')
axs[0].plot(totSeq_4, zSeq_4/zi, label='palm-tot', marker='', markersize=1, linestyle='-', linewidth=1.0, color='b')
axs[0].set_ylim(0.0,1.0)
axs[0].set_xlabel(r'$\mathrm{e}$ $(\mathrm{m^2/s^2})$', fontsize=12)
axs[0].set_ylabel(r'$\mathrm{z_i}$', fontsize=12)
axs[0].grid()
# axs[0].legend(loc='upper right', bbox_to_anchor=(0.9,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

axs[1].plot(funcs.flt_seq(rsvSeq_0[0::3]/totSeq_0[0::3]*100,0), zSeq_0[0::3]/zi, label='sowfa', marker='', markersize=1, linestyle='-', linewidth=1.0, color='r')
axs[1].plot(rsvSeq_4/totSeq_4*100, zSeq_4/zi, label='palm', marker='', markersize=1, linestyle='-', linewidth=1.0, color='b')
axs[1].set_xlim(60.0,100.0)
axs[1].set_ylim(0.0,1.0); axs[1].set_yticklabels([])
axs[1].set_xlabel(r'$\mathrm{e_{rsv}/e_{tot}}$ (%)', fontsize=12)
axs[1].grid()
# axs[1].legend(loc='upper left', bbox_to_anchor=(0.1,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

handles, labels = axs[0].get_legend_handles_labels()
lgdord = [0,3,1,4,2,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=3, mode='None', borderaxespad=0, fontsize=12)

saveDir = '/scratch/projects/deepwind/photo/profiles'
saveName = 'TKE_pr.png'
# plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
