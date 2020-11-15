import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt

# the directory where the wake data locate
prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_NBL_U10'
suffix = '_gs20'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M01'

var = 'u'
varD = 0 # u:0, v:1, w:2
varName = 'u'
varUnit = 'm/s'


cycle_no_list = ['.005'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
varSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

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
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

plotDataList = []
HList = []

for zInd in range(zNum):
    HList.append(zSeq[zInd])

    varData = varSeq[:,zInd,:,:]

    varNum = varData.size
    varData = varData.reshape(varNum)
    varMin = varData.min()
    varMax = varData.max()

    mean_ = varData.mean()
    var_ = varData.var()
    skw_ = skew(varData)
    krt_ = kurtosis(varData)

    binMin = np.floor(varMin)
    binMax = np.ceil(varMax)
    binNum = 100
    binWidth = (binMax - binMin) / binNum

    counts, bins = np.histogram(varData, bins=binNum, range=(binMin,binMax), density=False)
    binCoor = bins[:-1] # + 0.5*binWidth

    plotData = (binCoor, counts, varNum, binNum, binMin, binMax, mean_, var_, skw_, krt_)
    plotDataList.append(plotData)

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



# ### single plot
# fig, ax0 = plt.subplots(figsize=(8,8))
#
# # ax0.hist(varData, bins=binNum, range=(binMin,binMax), density=True, histtype='step', label='Probability Density')
# # ax0.hist(varData, bins=binNum, range=(binMin,binMax), density=True, histtype='step', cumulative=True, label='Cumulative Distribution')
# color = 'g'
# ax0.step(binCoor, counts/varNum, where='post', color=color)
# ax0.set_xlim(binMin, binMax)
# ax0.set_xlabel(var + ' (' + varUnit + ')', fontsize=12)
# ax0.set_ylabel('Probability Distribution', fontsize=12)
# ax0.tick_params(axis='y', labelcolor=color)
# # ax0.set_ylim(0, 1.06*binWidth)
#
# color = 'b'
# ax1 = ax0.twinx()
# ax1.step(binCoor, np.cumsum(counts/varNum), where='post', color=color)
# ax1.set_ylabel('Cumulative Distribution', fontsize=12)
# ax1.tick_params(axis='y', labelcolor=color)
#
#
# # plt.grid()
# ax0.text(0.06, 0.92, 'sample number: ' + str(varNum), transform=ax0.transAxes, fontdict={'size':12})
# ax0.text(0.06, 0.86, 'mean :' + str(round(mean_,4)), transform=ax0.transAxes, fontdict={'size':12})
# ax0.text(0.06, 0.80, 'variance :' + str(round(var_,4)), transform=ax0.transAxes, fontdict={'size':12})
# ax0.text(0.06, 0.74, 'skew :' + str(round(skw_,4)), transform=ax0.transAxes, fontdict={'size':12})
# ax0.text(0.06, 0.68, 'kurtosis :' + str(round(krt_,4)), transform=ax0.transAxes, fontdict={'size':12})
#
# saveDir = '/scratch/sowfadata/pp/' + jobName + '/'
# saveName = 'statistics' + '.png'
# # plt.savefig(saveDir + saveName)
# plt.show()
