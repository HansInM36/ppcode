import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
from netCDF4 import Dataset
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

    TIuSeq = 100 * np.power(uuSeq[:,1:], 0.5) / uSeq[:,1:] # somehow negative var2 values appear in the first two time steps; TI(z=0) should be specified to 0
    TIvSeq = 100 * np.power(vvSeq[:,1:], 0.5) / uSeq[:,1:]
    TIwSeq = 100 * np.power(wwSeq[:,1:], 0.5) / uSeq[:,1:]

    return tSeq, zSeq[1:], TIuSeq, TIvSeq, TIwSeq

def TI_pr_av_seq(tplot_para, tSeq, tDelta, zNum, TIuSeq, TIvSeq, TIwSeq):
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
def TI_pr_av(tplot_para, tSeq, zNum, TIuSeq, TIvSeq, TIwSeq):
    ave_itv = tplot_para[0]
    tplot = tplot_para[1]

    tmp = tSeq - tplot + ave_itv
    tInd_start = 0
    for tInd in range(tSeq.size):
        if tmp[tInd] < 0:
            tInd_start += 1
        else:
            break

    # compute the averaged velocity at a certain time and height
    TIuplotList = []
    TIvplotList = []
    TIwplotList = []
    for tInd in range(tSeq.size-tInd_start):
        TIuplotList.append(TIuSeq[tInd+tInd_start,:])
        TIwplotList.append(TIvSeq[tInd+tInd_start,:])
        TIvplotList.append(TIwSeq[tInd+tInd_start,:])
    TIuSeq = np.average(np.array(TIuplotList),axis=0)
    TIvSeq = np.average(np.array(TIvplotList),axis=0)
    TIwSeq = np.average(np.array(TIwplotList),axis=0)
    return tplot, TIuSeq, TIvSeq, TIwSeq

def ITP(varSeq, zSeq, z):
    f = interp1d(zSeq, varSeq, kind='linear')
    return f(z)

prjName = 'deepwind'
jobName_0 = 'gs10_refined'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_sowfa(dir_0, ((0,0,0),30.0))
t_seq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_av_seq((3600,151200,151200,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, TIuSeq_0, TIvSeq_0, TIwSeq_0)

# tplot, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_av((1200,146400), tSeq_0, zSeq_0.size, TIuSeq_0, TIvSeq_0, TIwSeq_0)

prjName = 'deepwind'
jobName_0 = 'gs10_0.0001'
dir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, zSeq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_sowfa(dir_0, ((0,0,0),30.0))
t_seq_0, TIuSeq_0, TIvSeq_0, TIwSeq_0 = TI_pr_av_seq((2400,146400,146400,1e6), tSeq_0, tSeq_0[-1]-tSeq_0[-2], zSeq_0.size, TIuSeq_0, TIvSeq_0, TIwSeq_0)

prjName = 'deepwind'
jobName_1 = 'gs10'
dir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, zSeq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_sowfa(dir_1, ((0,0,0),30.0))
t_seq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_av_seq((2400,146400,146400,1e6), tSeq_1, tSeq_1[-1]-tSeq_1[-2], zSeq_1.size, TIuSeq_1, TIvSeq_1, TIwSeq_1)

prjName = 'deepwind'
jobName_2 = 'gs10_0.01'
dir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, zSeq_2, TIuSeq_2, TIvSeq_2, TIwSeq_2 = TI_pr_sowfa(dir_2, ((0,0,0),30.0))
t_seq_2, TIuSeq_2, TIvSeq_2, TIwSeq_2 = TI_pr_av_seq((2400,74400,74400,1e6), tSeq_2, tSeq_2[-1]-tSeq_2[-2], zSeq_2.size, TIuSeq_2, TIvSeq_2, TIwSeq_2)



# prjDir = '/scratch/palmdata/JOBS'
# jobName_1  = 'deepwind_NBL'
# dir_1 = prjDir + '/' + jobName_1
# tSeq_1, zSeq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_palm(dir_1, jobName_1, ['.000', '.001'])

prjDir = '/scratch/palmdata/JOBS'
jobName_3  = 'deepwind_gs5_0.0001'
dir_3 = prjDir + '/' + jobName_3
tSeq_3, zSeq_3, TIuSeq_3, TIvSeq_3, TIwSeq_3 = TI_pr_palm(dir_3, jobName_3, ['.004','.005'])
TIuSeq_3, TIvSeq_3, TIwSeq_3 = TIuSeq_3[-1], TIvSeq_3[-1], TIwSeq_3[-1]


prjDir = '/scratch/palmdata/JOBS'
jobName_4  = 'deepwind_gs5_mdf'
dir_4 = prjDir + '/' + jobName_4
tSeq_4, zSeq_4, TIuSeq_4, TIvSeq_4, TIwSeq_4 = TI_pr_palm(dir_4, jobName_4, ['.004']) # ['.000','.001']
TIuSeq_4, TIvSeq_4, TIwSeq_4 = TIuSeq_4[-1], TIvSeq_4[-1], TIwSeq_4[-1]


prjDir = '/scratch/palmdata/JOBS'
jobName_5  = 'deepwind_gs5_main'
dir_5 = prjDir + '/' + jobName_5
tSeq_5, zSeq_5, TIuSeq_5, TIvSeq_5, TIwSeq_5 = TI_pr_palm(dir_5, jobName_5, ['.001']) # ['.000','.001']
TIuSeq_5, TIvSeq_5, TIwSeq_5 = TIuSeq_5[-1], TIvSeq_5[-1], TIwSeq_5[-1]

prjDir = '/scratch/palmdata/JOBS'
jobName_6  = 'deepwind_NBL_main'
dir_6 = prjDir + '/' + jobName_6
tSeq_6, zSeq_6, TIuSeq_6, TIvSeq_6, TIwSeq_6 = TI_pr_palm(dir_6, jobName_6, ['.004']) # ['.000','.001']
TIuSeq_6, TIvSeq_6, TIwSeq_6 = TIuSeq_6[0], TIvSeq_6[0], TIwSeq_6[0]



### plot
fig, ax = plt.subplots(figsize=(3,4.5))
fltw = 1
# plt.plot(funcs.flt_seq(TIuSeq_0[-1],fltw), zSeq_0, label='TIu-sowfa', linewidth=1.0, linestyle='-', color='r')
# plt.plot(funcs.flt_seq(TIvSeq_0[-1],fltw), zSeq_0, label='TIv-sowfa', linewidth=1.0, linestyle='-', color='b')
# plt.plot(funcs.flt_seq(TIwSeq_0[-1,::3],fltw), zSeq_0[::3], label='TIw-sowfa', linewidth=1.0, linestyle='-', color='g')
plt.plot(TIuSeq_4, zSeq_4, label='TIu-palm-pcr', linewidth=1.0, linestyle='--', color='r')
plt.plot(TIvSeq_4, zSeq_4, label='TIv-palm-pcr', linewidth=1.0, linestyle='--', color='b')
plt.plot(TIwSeq_4, zSeq_4, label='TIw-palm-pcr', linewidth=1.0, linestyle='--', color='g')
plt.plot(TIuSeq_5, zSeq_5, label='TIu-palm-main', linewidth=1.0, linestyle=':', color='r')
plt.plot(TIvSeq_5, zSeq_5, label='TIv-palm-main', linewidth=1.0, linestyle=':', color='b')
plt.plot(TIwSeq_5, zSeq_5, label='TIw-palm-main', linewidth=1.0, linestyle=':', color='g')
plt.xlabel('TI (%)', fontsize=12)
plt.ylabel('z (m)', fontsize=12)
# xaxis_min = 0
# xaxis_max = 12
# xaxis_d = 2
# yaxis_min = 0
# yaxis_max = 1000.0
# yaxis_d = 100.0
# plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.2,0.75), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'TI_pr.png'
# plt.savefig('/scratch/projects/deepwind/photo/profiles' + '/' + saveName)
plt.show()
plt.close()


### print
print('sowfa-0.0001 20m: ', np.round(ITP(TIuSeq_0, zSeq_0, 20),3), np.round(ITP(TIvSeq_0, zSeq_0, 20),3), np.round(ITP(TIwSeq_0, zSeq_0, 20),3))
print('sowfa-0.0001 100m: ', np.round(ITP(TIuSeq_0, zSeq_0, 100),3), np.round(ITP(TIvSeq_0, zSeq_0, 100),3), np.round(ITP(TIwSeq_0, zSeq_0, 100),3))
print('sowfa-0.0001 180m: ', np.round(ITP(TIuSeq_0, zSeq_0, 180),3), np.round(ITP(TIvSeq_0, zSeq_0, 180),3), np.round(ITP(TIwSeq_0, zSeq_0, 180),3))
print('sowfa-0.001 20m: ', np.round(ITP(TIuSeq_1, zSeq_1, 20),3), np.round(ITP(TIvSeq_1, zSeq_1, 20),3), np.round(ITP(TIwSeq_1, zSeq_1, 20),3))
print('sowfa-0.001 100m: ', np.round(ITP(TIuSeq_1, zSeq_1, 100),3), np.round(ITP(TIvSeq_1, zSeq_1, 100),3), np.round(ITP(TIwSeq_1, zSeq_1, 100),3))
print('sowfa-0.001 180m: ', np.round(ITP(TIuSeq_1, zSeq_1, 180),3), np.round(ITP(TIvSeq_1, zSeq_1, 180),3), np.round(ITP(TIwSeq_1, zSeq_1, 180),3))
print('sowfa-0.01 20m: ', np.round(ITP(TIuSeq_2, zSeq_2, 20),3), np.round(ITP(TIvSeq_2, zSeq_2, 20),3), np.round(ITP(TIwSeq_2, zSeq_2, 20),3))
print('sowfa-0.01 100m: ', np.round(ITP(TIuSeq_2, zSeq_2, 100),3), np.round(ITP(TIvSeq_2, zSeq_2, 100),3), np.round(ITP(TIwSeq_2, zSeq_2, 100),3))
print('sowfa-0.01 180m: ', np.round(ITP(TIuSeq_2, zSeq_2, 180),3), np.round(ITP(TIvSeq_2, zSeq_2, 180),3), np.round(ITP(TIwSeq_2, zSeq_2, 180),3))

print('palm-0.0001 20m: ', np.round(ITP(TIuSeq_3, zSeq_3, 20),3), np.round(ITP(TIvSeq_3, zSeq_3, 20),3), np.round(ITP(TIwSeq_3, zSeq_3, 20),3))
print('palm-0.0001 100m: ', np.round(ITP(TIuSeq_3, zSeq_3, 100),3), np.round(ITP(TIvSeq_3, zSeq_3, 100),3), np.round(ITP(TIwSeq_3, zSeq_3, 100),3))
print('palm-0.0001 180m: ', np.round(ITP(TIuSeq_3, zSeq_3, 180),3), np.round(ITP(TIvSeq_3, zSeq_3, 180),3), np.round(ITP(TIwSeq_3, zSeq_3, 180),3))
print('palm-0.001 20m: ', np.round(ITP(TIuSeq_4, zSeq_4, 20),3), np.round(ITP(TIvSeq_4, zSeq_4, 20),3), np.round(ITP(TIwSeq_4, zSeq_4, 20),3))
print('palm-0.001 100m: ', np.round(ITP(TIuSeq_4, zSeq_4, 100),3), np.round(ITP(TIvSeq_4, zSeq_4, 100),3), np.round(ITP(TIwSeq_4, zSeq_4, 100),3))
print('palm-0.001 180m: ', np.round(ITP(TIuSeq_4, zSeq_4, 180),3), np.round(ITP(TIvSeq_4, zSeq_4, 180),3), np.round(ITP(TIwSeq_4, zSeq_4, 180),3))
print('palm-0.01 20m: ', np.round(ITP(TIuSeq_5, zSeq_5, 20),3), np.round(ITP(TIvSeq_5, zSeq_5, 20),3), np.round(ITP(TIwSeq_5, zSeq_5, 20),3))
print('palm-0.01 100m: ', np.round(ITP(TIuSeq_5, zSeq_5, 100),3), np.round(ITP(TIvSeq_5, zSeq_5, 100),3), np.round(ITP(TIwSeq_5, zSeq_5, 100),3))
print('palm-0.01 180m: ', np.round(ITP(TIuSeq_5, zSeq_5, 180),3), np.round(ITP(TIvSeq_5, zSeq_5, 180),3), np.round(ITP(TIwSeq_5, zSeq_5, 180),3))
