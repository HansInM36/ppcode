import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.stats import kurtosis
from scipy.stats import skew
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

def sttts(varData):
    varNum = varData.size
    varData = varData.reshape(varNum)

    mean_ = varData.mean()

    varData -= mean_

    varMin = varData.min()
    varMax = varData.max()

    var_ = varData.var()
    skw_ = skew(varData)
    krt_ = kurtosis(varData)

    binMin = -2.0 # np.floor(varMin)
    binMax = 2.0 # np.ceil(varMax)
    binNum = 100
    binWidth = (binMax - binMin) / binNum

    counts, bins = np.histogram(varData, bins=binNum, range=(binMin,binMax), density=False)
    binCoor = bins[:-1] # + 0.5*binWidth

    return (binCoor, counts, varNum, binNum, binMin, binMax, mean_, var_, skw_, krt_)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq, xSeq, ySeq, zSeq, uSeq, coors = getData_sowfa(ppDir, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq, xSeq, ySeq, zSeq, vSeq, coors = getData_sowfa(ppDir, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq, xSeq, ySeq, zSeq, wSeq, coors = getData_sowfa(ppDir, 'prbg0', ((0,0,0),30.0), 'U', 2)
zInd = 0
pInd_start = xSeq.size*ySeq.size*zInd
pInd_end = xSeq.size*ySeq.size*(zInd+1)
uSeq = uSeq[pInd_start:pInd_end]
vSeq = vSeq[pInd_start:pInd_end]
wSeq = wSeq[pInd_start:pInd_end]
uData_0 = sttts(uSeq)
vData_0 = sttts(vSeq)
wData_0 = sttts(wSeq)

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs10'
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, uSeq = getData_palm(dir, jobName, 'M03', ['.022'], 'u')
tSeq, xSeq, ySeq, zSeq, vSeq = getData_palm(dir, jobName, 'M03', ['.022'], 'v')
tSeq, xSeq, ySeq, zSeq, wSeq = getData_palm(dir, jobName, 'M03', ['.022'], 'w')
zInd = 0
uSeq = uSeq[:,zInd,:,:]
vSeq = vSeq[:,zInd,:,:]
wSeq = wSeq[:,zInd,:,:]
uData_1 = sttts(uSeq)
vData_1 = sttts(vSeq)
wData_1 = sttts(wSeq)


""" statistics of u v w at certain height (sowfa vs palm) """
# fig, axs = plt.subplots(1,3, constrained_layout=False)
fig, axs = plt.subplots(1,3, constrained_layout=True)
fig.set_figwidth(9)
fig.set_figheight(3)

# u
bin_0, bin_1 = uData_0[0], uData_1[0]
counts_0, counts_1 = uData_0[1], uData_1[1]
varNum_0, varNum_1 = uData_0[2], uData_1[2]
binNum_0, binNum_1 = uData_0[3], uData_1[3]
binMin_0, binMin_1 = uData_0[4], uData_1[4]
binMax_0, binMax_1 = uData_0[5], uData_1[5]
mean_0, mean_1 = uData_0[6], uData_1[6]
var_0, var_1 = uData_0[7], uData_1[7]

axs[0].text(0.06, 0.88, 'mean-sowfa : ' + str(round(mean_0,2)), transform=axs[0].transAxes, fontsize=12)
axs[0].text(0.06, 0.78, 'mean-palm : ' + str(round(mean_1,2)), transform=axs[0].transAxes, fontsize=12)
axs[0].text(0.06, 0.68, 'variance-sowfa : ' + str(round(var_0,2)), transform=axs[0].transAxes, fontsize=12)
axs[0].text(0.06, 0.58, 'variance-palm : ' + str(round(var_1,2)), transform=axs[0].transAxes, fontsize=12)
axs[0].step(bin_0, counts_0/varNum_0, where='post', linestyle='-', color='red', label='sowfa')
axs[0].step(bin_1, counts_1/varNum_1, where='post', linestyle='-', color='orange', label='palm')
axs[0].set_xlim(binMin_0, binMax_0)
axs[0].set_ylim(0, 0.1)
axs[0].set_xticks([])
# axs[0].set_xlabel(r"$\mathrm{u}'$ (m/s)", fontsize=12)
axs[0].set_ylabel('Probability Distribution', fontsize=12)
axs[0].legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
# axs[0].grid()

# v
bin_0, bin_1 = vData_0[0], vData_1[0]
counts_0, counts_1 = vData_0[1], vData_1[1]
varNum_0, varNum_1 = vData_0[2], vData_1[2]
binNum_0, binNum_1 = vData_0[3], vData_1[3]
binMin_0, binMin_1 = vData_0[4], vData_1[4]
binMax_0, binMax_1 = vData_0[5], vData_1[5]
mean_0, mean_1 = vData_0[6], vData_1[6]
var_0, var_1 = vData_0[7], vData_1[7]

axs[1].text(0.06, 0.88, 'mean-sowfa : ' + str(round(mean_0,2)), transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.06, 0.78, 'mean-palm : ' + str(round(mean_1,2)), transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.06, 0.68, 'variance-sowfa : ' + str(round(var_0,2)), transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.06, 0.58, 'variance-palm : ' + str(round(var_1,2)), transform=axs[1].transAxes, fontsize=12)
axs[1].step(bin_0, counts_0/varNum_0, where='post', linestyle='-', color='blue', label='sowfa')
axs[1].step(bin_1, counts_1/varNum_1, where='post', linestyle='-', color='purple', label='palm')
axs[1].set_xlim(binMin_0, binMax_0)
axs[1].set_ylim(0, 0.1)
axs[1].set_xticks([])
axs[1].set_yticks([])
# axs[1].set_xlabel(r"$\mathrm{v}'$ (m/s)", fontsize=12)
axs[1].legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
# axs[1].grid()

# w
bin_0, bin_1 = wData_0[0], wData_1[0]
counts_0, counts_1 = wData_0[1], wData_1[1]
varNum_0, varNum_1 = wData_0[2], wData_1[2]
binNum_0, binNum_1 = wData_0[3], wData_1[3]
binMin_0, binMin_1 = wData_0[4], wData_1[4]
binMax_0, binMax_1 = wData_0[5], wData_1[5]
mean_0, mean_1 = wData_0[6], wData_1[6]
var_0, var_1 = wData_0[7], wData_1[7]

axs[2].text(0.06, 0.88, 'mean-sowfa : ' + str(round(mean_0,2)), transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.06, 0.78, 'mean-palm : ' + str(round(mean_1,2)), transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.06, 0.68, 'variance-sowfa : ' + str(round(var_0,2)), transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.06, 0.58, 'variance-palm : ' + str(round(var_1,2)), transform=axs[2].transAxes, fontsize=12)
axs[2].step(bin_0, counts_0/varNum_0, where='post', linestyle='-', color='green', label='sowfa')
axs[2].step(bin_1, counts_1/varNum_1, where='post', linestyle='-', color='turquoise', label='palm')
axs[2].set_xlim(binMin_0, binMax_0)
axs[2].set_ylim(0, 0.1)
axs[2].set_xticks([])
axs[2].set_yticks([])
# axs[2].set_xlabel(r"$\mathrm{w}'$ (m/s)", fontsize=12)
axs[2].legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
# axs[2].grid()
plt.show()


""" single plot """
fig, ax = plt.subplots(figsize=(6,4))

bin_ = plotData[0]
counts_ = plotData[1]
varNum_ = plotData[2]
binNum_ = plotData[3]
binMin_ = plotData[4]
binMax_ = plotData[5]
mean_ = plotData[6]
var_ = plotData[7]
skw_ = plotData[8]
krt_ = plotData[9]

ax.text(0.06, 0.88, 'sample number: ', transform=ax.transAxes, fontsize=10)
ax.text(0.06, 0.78, str(varNum_), transform=ax.transAxes, fontsize=10)
ax.text(0.06, 0.68, 'mean :' + str(round(mean_,2)), transform=ax.transAxes, fontsize=10)
ax.text(0.06, 0.58, 'variance :' + str(round(var_,2)), transform=ax.transAxes, fontsize=10)
ax.text(0.06, 0.48, 'skew :' + str(round(skw_,2)), transform=ax.transAxes, fontsize=10)
ax.text(0.06, 0.38, 'kurtosis :' + str(round(krt_,2)), transform=ax.transAxes, fontsize=10)

ax.step(bin_, counts_/varNum_, where='post', linestyle='-', color='firebrick')
ax.set_xlim(binMin_, binMax_)
ax.tick_params(axis='y', labelcolor='firebrick')
ax.set_xlabel('u (m/s)', fontsize=12)
ax.set_ylabel('Probability Distribution', color='firebrick', fontsize=12)
ax1 = ax.twinx()
ax1.step(bin_, np.cumsum(counts_/varNum_), where='post', linestyle='-', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('Cumulative Distribution', color='blue', fontsize=12)
plt.show()


### group plot
rNum, cNum = (4,2)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
fig.set_figwidth(8)
fig.set_figheight(12)

for i in range(zNum):
    rNo = int(np.floor(i/cNum))
    cNo = int(i - rNo*cNum)
    bin_ = plotDataList[i][0]
    counts_ = plotDataList[i][1]
    varNum_ = plotDataList[i][2]
    binNum_ = plotDataList[i][3]
    binMin_ = plotDataList[i][4]
    binMax_ = plotDataList[i][5]
    mean_ = plotDataList[i][6]
    var_ = plotDataList[i][7]
    skw_ = plotDataList[i][8]
    krt_ = plotDataList[i][9]

    color0 = 'g'
    color1 = 'b'


    axs[rNo,cNo].step(bin_, counts_/varNum_, where='post', linestyle=':', color=color0)
    axs[rNo,cNo].set_xlim(binMin_, binMax_)
    axs[rNo,cNo].tick_params(axis='y', labelcolor=color0)
    # if rNo != rNum - 1:
    #     axs[rNo,cNo].set_xticks([])
    # if cNo != 0:
    #     axs[rNo,cNo].set_yticks([])

    color = 'b'
    ax1 = axs[rNo,cNo].twinx()
    ax1.step(bin_, np.cumsum(counts_/varNum_), where='post', linestyle='-', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    if cNo != cNum - 1:
        ax1.set_yticks([])

    axs[rNo,cNo].text(0.06, 0.88, 'sample number: ', transform=axs[rNo,cNo].transAxes, fontsize=10)
    axs[rNo,cNo].text(0.06, 0.78, str(varNum_), transform=axs[rNo,cNo].transAxes, fontsize=10)
    axs[rNo,cNo].text(0.06, 0.68, 'mean :' + str(round(mean_,2)), transform=axs[rNo,cNo].transAxes, fontsize=10)
    axs[rNo,cNo].text(0.06, 0.58, 'variance :' + str(round(var_,2)), transform=axs[rNo,cNo].transAxes, fontsize=10)
    axs[rNo,cNo].text(0.06, 0.48, 'skew :' + str(round(skw_,2)), transform=axs[rNo,cNo].transAxes, fontsize=10)
    axs[rNo,cNo].text(0.06, 0.38, 'kurtosis :' + str(round(krt_,2)), transform=axs[rNo,cNo].transAxes, fontsize=10)

    axs[rNo,cNo].text(0.72, 0.88, 'h = ' + str(int(HList[i])) + 'm', transform=axs[rNo,cNo].transAxes, fontsize=10)

fig.text(0.5, 0.06, varName + ' (' + varUnit + ')', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Probability Distribution', va='center', rotation='vertical', fontsize=12, color=color0)
fig.text(0.96, 0.5, 'Cumulative Distribution', va='center', rotation='vertical', fontsize=12, color=color1)

fig.suptitle('')
# fig.tight_layout() # adjust the layout
saveName = 'statistics' + '.png'
plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
