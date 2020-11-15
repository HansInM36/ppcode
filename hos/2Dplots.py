import sys
sys.path.append('/scratch/ppcode/HOS')
import importlib as imp
import numpy as np
import math
from numpy import fft
import pickle
import matplotlib.pyplot as plt

import statistics as stt
import mathtool as mt
import waveClass as wvc

prjname = "regular"
jobname = "sstvt.org"
suffix = [".256_64.3.7", ".512_128.3.7", ".256_64.2.7", ".256_64.1.7", ".128_32.3.7", ".256_64.3.8", ".256_64.3.4"]
# suffix = [""]
sffNum = len(suffix)

data = []
for i in range(sffNum):
    readDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/data/'
    readName = "2Ddata_Dict" + suffix[i]
    fr = open(readDir + readName, 'rb')
    data.append(pickle.load(fr))
    fr.close()



### plot the surface elevation of y = constant at a certain time
tind = -3
yind = 16

x = []
y = []
for i in range(sffNum):
    x.append(data[i]['x'])
    y.append(data[i]['se'][tind][yind,:])

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,sffNum))
# colors = ['k']
# linestyles = ['-']
labels = ['case' + str(i) for i in range(sffNum)]

# for i in range(sffNum):
for i in [0,4,5,6]:
    axs.plot(x[i], y[i], linewidth=1.0, color=colors[i], label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel('SE (m)', fontsize=12)
axs.grid()
axs.set_xlabel('x (m)', fontsize=12)
axs.set_title('Surface Elevation y =' + str(data[0]['y'][yind]))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/'
saveName = 'SE_x' + '.png'
# plt.savefig(saveDir + saveName)
plt.show()


### plot the surface elevation time series of x, y = constant
xind = 16
yind = 16

x = []
y = []
for i in range(sffNum):
    x.append(data[i]['time'])
    y.append(data[i]['se'][:,yind,xind])

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,sffNum))
# colors = ['k']
# linestyles = ['-']
labels = ['case' + str(i) for i in range(sffNum)]

# ylims = [(1.26,1.27)]
for i in range(sffNum):
    axs.plot(x[i], y[i], linewidth=1.0, color=colors[i], label=labels[i])
axs.set_xlim(0,20)
# axs.set_ylim(ylims[0])
axs.set_ylabel('SE (m)', fontsize=12)
axs.grid()
axs.set_xlabel('t (s)', fontsize=12)
axs.set_title('Surface Elevation Time Series')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0)
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/'
saveName = 'SE_ts' + ".png"
plt.savefig(saveDir + saveName)
plt.show()


### check wave data of a certain timestep
caseNo = 0
tind = -3
yind = 16
x = data[caseNo]['x']
y = data[caseNo]['eta'][tind][yind,:]
wave = wvc.count_wave(x, y)
print("lambda = ", np.mean(wave[0]))
print("k = ", 2*np.pi / np.mean(wave[0]))

### check wave data of a certain position
xind = 16
yind = 16
x = data[caseNo]['time']
y = data[caseNo]['eta'][:,yind,xind]
wave = wvc.count_wave(x, y)
print("T = ", np.mean(wave[0]))
print("var_T = ", np.var(wave[0]))
print("H = ", np.mean(wave[1]))
print("var_H = ", np.var(wave[1]))
print("omega = ", 2*np.pi / np.mean(wave[0]))
