import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
import sliceDataClass as sdc
import funcs
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

# sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7',]
sliceList = ['Nz2']
sliceNum = len(sliceList)

# coordinate transmation
O = (0,0,0)
alpha = 30.0

var = 'U'
varD = 0 # u:0, v:1, w:2
varNameDict = {0: 'u', 1:'v', 2:'w'}

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']
tInd = -1 # index of time step

plotDataList = []
HList = []

for slice in sliceList:
    readDir = ppDir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    slc = sdc.Slice(data_org, 2)
    H = slc.N_location # height of this plane
    HList.append(H)
    plotData = slc.meshITP_Nz((0,2560,256), (0,2560,256), slc.data[var][tInd][:,varD], method_='linear')
    plotDataList.append(plotData)

### group plot
rNum, cNum = (4,2)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=True)

fig.set_figwidth(8)
fig.set_figheight(12)

cbreso = 100 # resolution of colorbar
vMin, vMax, vDelta = (-2.4, 2.4, 0.4)
levels = np.linspace(vMin, vMax, cbreso + 1)

for i in range(sliceNum):
    rNo = int(np.floor(i/cNum))
    cNo = int(i - rNo*cNum)
    x_ = plotDataList[i][0]
    y_ = plotDataList[i][1]
    v_ = plotDataList[i][2]
    # clb is for creating a common colorbar
    clb = axs[rNo,cNo].contourf(x_, y_, v_ - v_.mean(), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    if rNo == rNum - 1:
        axs[rNo,cNo].set_xlabel('x (m)', fontsize=12)
    else:
        axs[rNo,cNo].set_xticks([])
    if cNo == 0:
        axs[rNo,cNo].set_ylabel('y (m)', fontsize=12)
    else:
        axs[rNo,cNo].set_yticks([])
    axs[rNo,cNo].text(0.6, 1.02, 'h = ' + str(int(HList[i])) + 'm', transform=axs[rNo,cNo].transAxes, fontsize=12)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = fig.colorbar(clb, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.set_label(varNameDict[varD] + ' (m/s)', fontsize=12)
cbar.ax.tick_params(labelsize=12)
fig.suptitle('t = ' + str(tSeq[tInd]) + 's')
saveName = varNameDict[varD] + '_contour_' + str(tSeq[tInd]) + '_' + sliceList[0] + '_etc' + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()



### single plot
sliceInd = 0
slice = sliceList[sliceInd]

plotDataList = []

readDir = ppDir + '/data/'
readName = slice
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

slc = sdc.Slice(data_org, 2)
H = slc.N_location # height of this plane

tIndList = list(range(0,600,10))

tNum = len(tIndList)

for tInd in tIndList:
    print(tInd)
    tmp = slc.data[var][tInd]
    tmp = funcs.trs(tmp,O,alpha) # coordinate transformation
    plotData = slc.meshITP_Nz((0,2000,100), (0,2000,100), slc.data[var][tInd][:,varD], method_='linear')
    plotDataList.append(plotData)


vMin, vMax, vDelta = (-1.8, 1.8, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

for tIndInd in range(tNum):
    fig, axs = plt.subplots(figsize=(8,8), constrained_layout=True)

    x_ = plotDataList[tIndInd][0]
    y_ = plotDataList[tIndInd][1]
    v_ = plotDataList[tIndInd][2]
    v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_ - v_.mean(), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tIndList[tIndInd]],2)) + 's', transform=axs.transAxes, fontsize=12)
    cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=12)
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.title('h = ' + str(int(H)) + 'm')
    # saveName = varNameDict[varD] + '_contour_' + str(tSeq[tInd]) + '_' + sliceList[zInd] + '.png'
    saveName = "%.4d" % tIndList[tIndInd] + '.png'
    # plt.savefig(ppDir + '/animation/' + saveName, bbox_inches='tight')
    plt.savefig(ppDir + '/animation/' + saveName)
    # plt.show()
    plt.close()
