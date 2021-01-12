import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt

def TI_pr_sowfa(dir, trs_para):
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]*np.pi/180

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
    uvSeq = aveData['uv_mean']
    wwSeq = aveData['ww_mean']
    uSeq = aveData['U_mean']
    vSeq = aveData['V_mean']

    uvarianceSeq = uuSeq*np.power(np.cos(alpha),2) + 2*uvSeq*np.cos(alpha)*np.sin(alpha) + vvSeq*np.power(np.sin(alpha),2)
    vvarianceSeq = uuSeq*np.power(np.sin(alpha),2) + 2*uvSeq*np.sin(alpha)*np.cos(alpha) + vvSeq*np.power(np.cos(alpha),2)
    wvarianceSeq = wwSeq
    umeanSeq = uSeq*np.cos(alpha) + vSeq*np.sin(alpha)
    TIuSeq = 100 * np.power(uvarianceSeq,0.5) / umeanSeq
    TIvSeq = 100 * np.power(vvarianceSeq,0.5) / umeanSeq
    TIwSeq = 100 * np.power(wvarianceSeq,0.5) / umeanSeq
    return tSeq, zSeq, TIuSeq, TIvSeq, TIwSeq
def TI_pr_palm(dir, jobName, run_no_list):
    """ extract horizontal average of velocity at various times and heights """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    uSeq_list = []
    uuSeq_list = []
    vSeq_list = []
    vvSeq_list = []
    wSeq_list = []
    wwSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        uSeq_list.append(np.array(nc_file_list[i].variables['u'][:], dtype=type(nc_file_list[i].variables['u'])))
        uuSeq_list.append(np.array(nc_file_list[i].variables['u*2'][:], dtype=type(nc_file_list[i].variables['u*2'])))
        vSeq_list.append(np.array(nc_file_list[i].variables['v'][:], dtype=type(nc_file_list[i].variables['v'])))
        vvSeq_list.append(np.array(nc_file_list[i].variables['v*2'][:], dtype=type(nc_file_list[i].variables['v*2'])))
        wSeq_list.append(np.array(nc_file_list[i].variables['w'][:], dtype=type(nc_file_list[i].variables['w'])))
        wwSeq_list.append(np.array(nc_file_list[i].variables['w*2'][:], dtype=type(nc_file_list[i].variables['w*2'])))

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables['u'].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    uSeq = np.concatenate([uSeq_list[i] for i in range(run_num)], axis=0)
    uSeq = uSeq.astype(float)
    uuSeq = np.concatenate([uuSeq_list[i] for i in range(run_num)], axis=0)
    uuSeq = uuSeq.astype(float)
    vSeq = np.concatenate([vSeq_list[i] for i in range(run_num)], axis=0)
    vSeq = vSeq.astype(float)
    vvSeq = np.concatenate([vvSeq_list[i] for i in range(run_num)], axis=0)
    vvSeq = vvSeq.astype(float)
    wSeq = np.concatenate([wSeq_list[i] for i in range(run_num)], axis=0)
    wSeq = wSeq.astype(float)
    wwSeq = np.concatenate([wwSeq_list[i] for i in range(run_num)], axis=0)
    wwSeq = wwSeq.astype(float)

    TIuSeq = 100 * np.power(uuSeq[2:,1:], 0.5) / uSeq[2:,1:] # somehow negative var2 values appear in the first two time steps; TI(z=0) should be specified to 0
    TIvSeq = 100 * np.power(vvSeq[2:,1:], 0.5) / uSeq[2:,1:]
    TIwSeq = 100 * np.power(wwSeq[2:,1:], 0.5) / uSeq[2:,1:]

    return tSeq, zSeq[1:], TIuSeq, TIvSeq, TIwSeq

def TI_pr_ave(tplot_para, tSeq, tDelta, zNum, TIuSeq, TIvSeq, TIwSeq):
    ave_itv = tplot_para[0]
    tplot_start = tplot_para[1]
    tplot_end = tplot_para[2]
    tplot_delta = tplot_para[3]
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

    # compute the averaged velocity at a certain time and height
    TIuplotList = []
    TIvplotList = []
    TIwplotList = []
    for tplot in tplotList:
        TIuplot = np.zeros(zNum)
        TIvplot = np.zeros(zNum)
        TIwplot = np.zeros(zNum)
        for zInd in range(zNum):
            f1 = interp1d(tSeq, TIuSeq[:,zInd], kind='linear', fill_value='extrapolate')
            f2 = interp1d(tSeq, TIvSeq[:,zInd], kind='linear', fill_value='extrapolate')
            f3 = interp1d(tSeq, TIwSeq[:,zInd], kind='linear', fill_value='extrapolate')
            tplot_tmp = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
            TIuplot[zInd] = f1(tplot_tmp).mean()
            TIvplot[zInd] = f2(tplot_tmp).mean()
            TIwplot[zInd] = f3(tplot_tmp).mean()
        TIuplotList.append(TIuplot)
        TIvplotList.append(TIvplot)
        TIwplotList.append(TIwplot)
    t_seq = np.array(tplotList)
    TIuSeq = np.array(TIuplotList)
    TIvSeq = np.array(TIvplotList)
    TIwSeq = np.array(TIwplotList)
    return t_seq, TIuSeq, TIvSeq, TIwSeq


prjName = 'deepwind'
jobName_0 = 'gs10'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_sowfa(dir_0, ((0,0,0),30.0))
t_seq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_ave((3600,144000,144000,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, TIuSeq_0, TIvSeq_0, TIwSeq_0)

prjName = 'deepwind'
jobName_0 = 'gs10_0.0001'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_sowfa(dir_0, ((0,0,0),30.0))
t_seq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_ave((3600,144000,144000,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, TIuSeq_0, TIvSeq_0, TIwSeq_0)



prjDir = '/scratch/palmdata/JOBS'
jobName_1  = 'deepwind_NBL'
dir_1 = prjDir + '/' + jobName_1
tSeq_1, zSeq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_palm(dir_1, jobName_1, ['.000', '.001'])


fig, ax = plt.subplots(figsize=(3,4.5))
plt.plot(TIuSeq_0[-1], zSeq_0, label='TIu-sowfa', linewidth=1.0, linestyle='-', color='r')
plt.plot(TIvSeq_0[-1], zSeq_0, label='TIv-sowfa', linewidth=1.0, linestyle='-', color='b')
plt.plot(TIwSeq_0[-1], zSeq_0, label='TIw-sowfa', linewidth=1.0, linestyle='-', color='g')
plt.plot(TIuSeq_1[-1], zSeq_1, label='TIu-palm', linewidth=1.0, linestyle='--', color='r')
plt.plot(TIvSeq_1[-1], zSeq_1, label='TIv-palm', linewidth=1.0, linestyle='--', color='b')
plt.plot(TIwSeq_1[-1], zSeq_1, label='TIw-palm', linewidth=1.0, linestyle='--', color='g')
plt.xlabel('TI', fontsize=12)
# plt.ylabel('z (m)', fontsize=12)
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), [], fontsize=12)
plt.legend(bbox_to_anchor=(0.4,0.75), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'TI_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()
