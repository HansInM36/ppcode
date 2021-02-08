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

def uwflux_sowfa(ppDir, trs_para, varD):
    O = trs_para[0]
    alpha = trs_para[1]
    # ave_itv = t_para[0]
    # tplot = t_para[1]

    fr = open(ppDir + '/data/' + 'aveData', 'rb')
    aveData = pickle.load(fr)
    fr.close()

    zSeq = aveData['H']
    zNum = zSeq.size

    tSeq = aveData['time']
    tNum = tSeq.size
    tDelta = tSeq[1] - tSeq[0]

    uwSeq = aveData['uw_mean']
    vwSeq = aveData['vw_mean']
    wwSeq = aveData['ww_mean']
    R13Seq = aveData['R13_mean']
    R23Seq = aveData['R23_mean']
    R33Seq = aveData['R33_mean']

    rsvSeq = np.zeros((tNum,zNum))
    sgsSeq = np.zeros((tNum,zNum))

    for zInd in range(zNum):
        tmp = np.concatenate((uwSeq[:,zInd].reshape(tNum,1), vwSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp = np.concatenate((tmp, wwSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp_ = funcs.trs(tmp,O,alpha)
        rsvSeq[:,zInd] = tmp_[:,varD]

        tmp = np.concatenate((R13Seq[:,zInd].reshape(tNum,1), R23Seq[:,zInd].reshape(tNum,1)), axis=1)
        tmp = np.concatenate((tmp, R33Seq[:,zInd].reshape(tNum,1)), axis=1)
        tmp_ = funcs.trs(tmp,O,alpha)
        sgsSeq[:,zInd] = tmp_[:,varD]

    totSeq = rsvSeq + sgsSeq
    return tSeq, zSeq, rsvSeq, sgsSeq, totSeq
def uwflux_plot_sowfa(uwSeq, tSeq, zNum, t_para):
    tDelta = tSeq[1] - tSeq[0]
    ave_itv = t_para[0]
    tplot = t_para[1]

    uwSeq_ = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, uwSeq[:,zInd], kind='linear', fill_value='extrapolate')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        uwSeq_[zInd] = f(tplotSeq).mean()
    return uwSeq_

def uwflux_palm(ppDir, run_no_list):
    run_num = len(run_no_list)

    rsv = 'w*u*'
    sgs = 'w"u"'

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    rsvSeq_list = []
    sgsSeq_list = []
    for i in range(run_num):
        input_file = prjDir + '/' + jobName + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        rsvSeq_list.append(np.array(nc_file_list[i].variables[rsv][:], dtype=type(nc_file_list[i].variables[rsv])))
        sgsSeq_list.append(np.array(nc_file_list[i].variables[sgs][:], dtype=type(nc_file_list[i].variables[sgs])))

    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['zu'].dimensions)) #list dimensions of a specified variable

    height = list(nc_file_list[0].variables[rsv].dimensions)[1] # the height name string
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
def uwflux_plot_palm(uwSeq, tSeq, zNum, tplot):
    uwSeq_ = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq[1:], uwSeq[1:,zInd], kind='linear', fill_value='extrapolate')
        uwSeq_[zInd] = f(tplot)
    return uwSeq_

def ITP(varSeq, zSeq, z):
    f = interp1d(zSeq, varSeq, kind='linear')
    return f(z)

""" SOWFA """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'

jobName = 'gs10_0.0001'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_0, zSeq_0, rsvSeq_0, sgsSeq_0, totSeq_0 = uwflux_sowfa(ppDir_0, ((0,0,0),30), 0)
totSeq_0 = uwflux_plot_sowfa(totSeq_0, tSeq_0, zSeq_0.size, (2400.0,146400.0))

jobName = 'gs10'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_1, zSeq_1, rsvSeq_1, sgsSeq_1, totSeq_1 = uwflux_sowfa(ppDir_1, ((0,0,0),30), 0)
totSeq_1 = uwflux_plot_sowfa(totSeq_1, tSeq_1, zSeq_1.size, (2400.0,146400.0))

jobName = 'gs10_0.01'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_2, zSeq_2, rsvSeq_2, sgsSeq_2, totSeq_2  = uwflux_sowfa(ppDir_2, ((0,0,0),30), 0)
totSeq_2 = uwflux_plot_sowfa(totSeq_2, tSeq_2, zSeq_2.size, (2400.0,74400.0))



""" PALM """
prjDir = '/scratch/palmdata/JOBS'

jobName  = 'deepwind_gs5_0.0001'
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_3, zSeq_3, rsvSeq_3, sgsSeq_3, totSeq_3 = uwflux_palm(ppDir, ['.004','.005'])
totSeq_3 = totSeq_3[-1]
# totSeq_3 = uwflux_plot_palm(totSeq_3, tSeq_3, zSeq_3.size, 79200.0)

jobName  = 'deepwind_gs5_mdf'
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_4, zSeq_4, rsvSeq_4, sgsSeq_4, totSeq_4 = uwflux_palm(ppDir, ['.003','.004'])
totSeq_4 = totSeq_4[-1]
# totSeq_4 = uwflux_plot_palm(totSeq_4, tSeq_4, zSeq_4.size, 79200.0)

jobName  = 'deepwind_gs5_0.01'
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq_5, zSeq_5, rsvSeq_5, sgsSeq_5, totSeq_5 = uwflux_palm(ppDir, ['.003','.004'])
totSeq_5 = totSeq_5[-1]
# totSeq_5 = uwflux_plot_palm(totSeq_5, tSeq_5, zSeq_5.size, 432000.0)

### plot
cflInd_sowfa = 9
cflInd_palm = 18
fig, ax = plt.subplots(figsize=(6,6))
plt.plot(totSeq_0[:cflInd_sowfa], zSeq_0[:cflInd_sowfa], label='z0=0.0001(sowfa)', linestyle='-', linewidth=1.0, color='r')
plt.plot(totSeq_1[:cflInd_sowfa], zSeq_1[:cflInd_sowfa], label='z0=0.001(sowfa)', linestyle='-', linewidth=1.0, color='b')
plt.plot(totSeq_2[:cflInd_sowfa], zSeq_2[:cflInd_sowfa], label='z0=0.01(sowfa)', linestyle='-', linewidth=1.0, color='g')
plt.plot(totSeq_3[:cflInd_palm], zSeq_3[:cflInd_palm], label='z0=0.0001(palm)', linestyle='--', linewidth=1.0, color='r')
plt.plot(totSeq_4[:cflInd_palm], zSeq_4[:cflInd_palm], label='z0=0.001(palm)', linestyle='--', linewidth=1.0, color='b')
plt.plot(totSeq_5[:cflInd_palm], zSeq_5[:cflInd_palm], label='z0=0.01(palm)', linestyle='--', linewidth=1.0, color='g')

plt.xlabel("u-component vertical momentum flux" + ' (' + r'$\mathrm{m^2/s^2}$' + ')')
plt.ylabel('z (m)')
xaxis_min = -0.16
xaxis_max = 0.0
xaxis_d = 0.04
yaxis_min = 0
yaxis_max = 180.0
yaxis_d = 20.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
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
