import os
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

def uwflux_sowfa(dir, trs_para, varD):
    O = trs_para[0]
    alpha = trs_para[1]
    # ave_itv = t_para[0]
    # tplot = t_para[1]

    fr = open(dir + '/data/' + 'aveData', 'rb')
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

def uwflux_palm(prjDir, jobName, run_no_list):
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

""" PALM """
jobName  = 'WRFPALM_20150701'
dir = '/scratch/palmdata/JOBS'
tSeq, zSeq, rsvSeq, sgsSeq, totSeq = uwflux_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'])



### plot
fig, ax = plt.subplots(figsize=(6,4.5))
colors = plt.cm.jet(np.linspace(0,1,tSeq.size))
for i in range(1,tSeq.size):
    plt.plot(totSeq[i], zSeq, label='t = ' + str(int(tSeq[i])) + 's', linewidth=1.0, color=colors[i])

plt.xlabel("u-component vertical momentum flux" + ' (' + r'$\mathrm{m^2/s^2}$' + ')')
plt.ylabel('z (m)')
xaxis_min = -0.02
xaxis_max = 0.12
xaxis_d = 0.02
yaxis_min = 0
yaxis_max = 400.0
yaxis_d = 50.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
#plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title(jobName)
fig.tight_layout() # adjust the layout
saveName = 'uwflux_pr.png'
saveDir = '/scratch/projects/WRFnesting/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
