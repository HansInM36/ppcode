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
rsvSeq_0 = TKE_plot_sowfa(rsvSeq_0, tSeq_0, zSeq_0.size, (2400.0,146400.0))
sgsSeq_0 = TKE_plot_sowfa(sgsSeq_0, tSeq_0, zSeq_0.size, (2400.0,146400.0))
totSeq_0 = TKE_plot_sowfa(totSeq_0, tSeq_0, zSeq_0.size, (2400.0,146400.0))

jobName = 'gs10'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_1, zSeq_1, rsvSeq_1, sgsSeq_1, totSeq_1 = TKE_sowfa(ppDir_1, ((0,0,0),30), 0)
rsvSeq_1 = TKE_plot_sowfa(rsvSeq_1, tSeq_1, zSeq_1.size, (2400.0,146400.0))
sgsSeq_1 = TKE_plot_sowfa(sgsSeq_1, tSeq_1, zSeq_1.size, (2400.0,146400.0))
totSeq_1 = TKE_plot_sowfa(totSeq_1, tSeq_1, zSeq_1.size, (2400.0,146400.0))

jobName = 'gs20'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_2, zSeq_2, rsvSeq_2, sgsSeq_2, totSeq_2 = TKE_sowfa(ppDir_2, ((0,0,0),30), 0)
rsvSeq_2 = TKE_plot_sowfa(rsvSeq_2, tSeq_2, zSeq_2.size, (2400.0,432000.0))
sgsSeq_2 = TKE_plot_sowfa(sgsSeq_2, tSeq_2, zSeq_2.size, (2400.0,432000.0))
totSeq_2 = TKE_plot_sowfa(totSeq_2, tSeq_2, zSeq_2.size, (2400.0,432000.0))

jobName = 'gs40'
ppDir_3 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_3, zSeq_3, rsvSeq_3, sgsSeq_3, totSeq_3 = TKE_sowfa(ppDir_3, ((0,0,0),30), 0)
rsvSeq_3 = TKE_plot_sowfa(rsvSeq_3, tSeq_3, zSeq_3.size, (2400.0,146400.0))
sgsSeq_3 = TKE_plot_sowfa(sgsSeq_3, tSeq_3, zSeq_3.size, (2400.0,146400.0))
totSeq_3 = TKE_plot_sowfa(totSeq_3, tSeq_3, zSeq_3.size, (2400.0,146400.0))



""" PALM """
prjDir = '/scratch/palmdata/JOBS'

jobName  = 'deepwind_gs5_0.0001_main'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq_4, zSeq_4, rsvSeq_4, sgsSeq_4, totSeq_4 = TKE_palm(dir, ['.000','.001'])
rsvSeq_4 = rsvSeq_4[-1]
sgsSeq_4 = sgsSeq_4[-1]
totSeq_4 = totSeq_4[-1]

jobName  = 'deepwind_gs10'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq_5, zSeq_5, rsvSeq_5, sgsSeq_5, totSeq_5 = TKE_palm(dir, ['.024','.025'])
rsvSeq_5 = rsvSeq_5[-1]
sgsSeq_5 = sgsSeq_5[-1]
totSeq_5 = totSeq_5[-1]

jobName  = 'deepwind_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_6, zSeq_6, rsvSeq_6, sgsSeq_6, totSeq_6 = TKE_palm(ppDir, ['.001','.002'])
rsvSeq_6 = rsvSeq_6[-1]
sgsSeq_6 = sgsSeq_6[-1]
totSeq_6 = totSeq_6[-1]

jobName  = 'deepwind_gs40'
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_7, zSeq_7, rsvSeq_7, sgsSeq_7, totSeq_7 = TKE_palm(ppDir, ['.000'])
rsvSeq_7 = rsvSeq_7[-1]
sgsSeq_7 = sgsSeq_7[-1]
totSeq_7 = totSeq_7[-1]



### plot
fig, ax = plt.subplots(figsize=(5,5))

zi=700
colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

plt.plot(funcs.flt_seq(rsvSeq_0[0::3]/totSeq_0[0::3]*100,0), zSeq_0[0::3]/zi, label='sowfa-gs5', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_sowfa[0])
# plt.plot(rsvSeq_1/totSeq_1*100, zSeq_1/zi, label='sowfa-gs10', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_sowfa[1])
# plt.plot(rsvSeq_2/totSeq_2*100, zSeq_2/zi, label='sowfa-gs20', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_sowfa[2])
# plt.plot(rsvSeq_3/totSeq_3*100, zSeq_3/zi, label='sowfa-gs40', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_sowfa[3])
plt.plot(rsvSeq_4/totSeq_4*100, zSeq_4/zi, label='palm-gs5', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_palm[0])
# plt.plot(rsvSeq_5/totSeq_5*100, zSeq_5/zi, label='palm-gs10', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_palm[1])
# plt.plot(rsvSeq_6/totSeq_6*100, zSeq_6/zi, label='palm-gs20', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_palm[2])
# plt.plot(rsvSeq_7/totSeq_7*100, zSeq_7/zi, label='palm-gs40', marker='', markersize=1, linestyle='-', linewidth=1.0, color=colors_palm[3])

plt.xlabel(r'$\mathrm{e_{rsv}/e_{tot}}$ (%)', fontsize=12)
plt.ylabel(r'$\mathrm{z_i}$ (m)', fontsize=12)
xaxis_min = 60.0
xaxis_max = 100.0
xaxis_d = 5.0
yaxis_min = 0.0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.06,0.78), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName_save + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()


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
axs[0].set_xlabel(r'$\mathrm{e}$', fontsize=12)
axs[0].set_ylabel(r'$\mathrm{z_i}$ (m)', fontsize=12)
axs[0].grid()
axs[0].legend(loc='upper right', bbox_to_anchor=(0.9,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

axs[1].plot(funcs.flt_seq(rsvSeq_0[0::3]/totSeq_0[0::3]*100,0), zSeq_0[0::3]/zi, label='sowfa', marker='', markersize=1, linestyle='-', linewidth=1.0, color='r')
axs[1].plot(rsvSeq_4/totSeq_4*100, zSeq_4/zi, label='palm', marker='', markersize=1, linestyle='-', linewidth=1.0, color='b')
axs[1].set_ylim(0.0,1.0)
axs[1].set_xlabel(r'$\mathrm{e_{rsv}/e_{tot}}$ (%)', fontsize=12)
axs[1].grid()
axs[1].legend(loc='upper left', bbox_to_anchor=(0.1,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName_save + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()


### print
print('sowfa-0.0001 20m: ', np.round(ITP(totSeq_0, zSeq_0, 20),3))
print('sowfa-0.0001 100m: ', np.round(ITP(totSeq_0, zSeq_0, 100),3))
print('sowfa-0.0001 180m: ', np.round(ITP(totSeq_0, zSeq_0, 180),3))
print('sowfa-0.001 20m: ', np.round(ITP(totSeq_1, zSeq_1, 20),3))
print('sowfa-0.001 100m: ', np.round(ITP(totSeq_1, zSeq_1, 100),3))
print('sowfa-0.001 180m: ', np.round(ITP(totSeq_1, zSeq_1, 180),3))
print('sowfa-0.01 20m: ', np.round(ITP(totSeq_2, zSeq_2, 20),3))
print('sowfa-0.01 100m: ', np.round(ITP(totSeq_2, zSeq_2, 100),3))
print('sowfa-0.01 180m: ', np.round(ITP(totSeq_2, zSeq_2, 180),3))

print('palm-0.0001 20m: ', np.round(ITP(totSeq_3, zSeq_3, 20),3))
print('palm-0.0001 100m: ', np.round(ITP(totSeq_3, zSeq_3, 100),3))
print('palm-0.0001 180m: ', np.round(ITP(totSeq_3, zSeq_3, 180),3))
print('palm-0.001 20m: ', np.round(ITP(totSeq_4, zSeq_4, 20),3))
print('palm-0.001 100m: ', np.round(ITP(totSeq_4, zSeq_4, 100),3))
print('palm-0.001 180m: ', np.round(ITP(totSeq_4, zSeq_4, 180),3))
print('palm-0.01 20m: ', np.round(ITP(totSeq_5, zSeq_5, 20),3))
print('palm-0.01 100m: ', np.round(ITP(totSeq_5, zSeq_5, 100),3))
print('palm-0.01 180m: ', np.round(ITP(totSeq_5, zSeq_5, 180),3))
