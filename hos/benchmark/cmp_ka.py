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
jobname = ["ka01", "ka02"]

# suffix = [".256_64.3.7", ".512_128.3.7", ".256_64.2.7", ".256_64.1.7", ".128_32.3.7", ".256_64.3.8", ".256_64.3.4"]
jobnum = len(jobname)

data = []
for i in range(jobnum):
    readDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname[i] + '/data/'
    readName = "VP_card_fitted_Dict"
    fr = open(readDir + readName, 'rb')
    data.append(pickle.load(fr))
    fr.close()

# print('time step: ', data[0]['time'], ' time-size: ', data[0]['time'].size)
# print('x-grids: ', data[0]['x'], ' x-size: ', data[0]['x'].size)
# print('y-grids: ', data[0]['y'], ' y-size: ', data[0]['y'].size)
# print('z-grids: ', data[0]['z'], ' z-size: ', data[0]['z'].size)

waveL = []
wavek = []
waveT = []
waveH = []
waveOmega = []

tind = -3
xind = 25
yind = 5

for i in range(jobnum):
    x = data[i]['x']
    y = data[i]['eta'][tind,0,yind,:]
    wave = wvc.count_wave(x, y)
    waveL.append(np.mean(wave[0]))
    wavek.append(2*np.pi / np.mean(wave[0]))

for i in range(jobnum):
    x = data[i]['time']
    y = data[i]['eta'][:,0,yind,xind]
    wave = wvc.count_wave(x, y)
    waveT.append(np.mean(wave[0]))
    waveOmega.append(2*np.pi / np.mean(wave[0]))
    waveH.append(np.mean(wave[1]))


# ### check wave data of a certain position
# xind = 16
# yind = 16
# x = data[caseNo]['time']
# y = data[caseNo]['eta'][:,yind,xind]
# wave = wvc.count_wave(x, y)
# print("T = ", np.mean(wave[0]))
# print("var_T = ", np.var(wave[0]))
# print("H = ", np.mean(wave[1]))
# print("var_H = ", np.var(wave[1]))
# print("omega = ", 2*np.pi / np.mean(wave[0]))




### plot the surface elevation of y = constant at a certain time
tind = -3
yind = 5

x = []
y = []
for i in range(jobnum):
    x.append(data[i]['x'])
    y.append(data[i]['eta'][tind,0,yind,:])

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(x[i], y[i], linewidth=1.0, color=colors[i], linestyle='-', marker=None, label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel(r'$\eta$ (m)', fontsize=12)
axs.grid()
axs.set_xlabel('x (m)', fontsize=12)
axs.set_title('Surface Elevation y =' + str(data[0]['y'][yind]))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveName = 'SE_x' + '.png'
plt.savefig(saveDir + saveName)
plt.show()



### plot the time series of surface elevation at position (xind, yind)
xind = 25
yind = 5

x = []
y = []
for i in range(jobnum):
    x.append(data[i]['time'])
    y.append(data[i]['eta'][:,0,yind,xind])

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(x[i], y[i], linewidth=1.0, color=colors[i], linestyle='-', marker=None, label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel(r'$\eta$ (m)', fontsize=12)
axs.grid()
axs.set_xlabel('time (s)', fontsize=12)
axs.set_title('Surface Elevation Time Series')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveName = 'SE_ts' + '.png'
plt.savefig(saveDir + saveName)
plt.show()



### plot the time series of u at position (xind, yind, zind)
xind = 25
yind = 5
zind = 24


x = []
y = []
for i in range(jobnum):
    print('z = ', data[i]['z'][zind])
    x.append(data[i]['time'])
    y.append(data[i]['u'][:,zind,yind,xind])

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(x[i], y[i], linewidth=1.0, color=colors[i], linestyle='-', marker=None, label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel('u (m/s)', fontsize=12)
axs.grid()
axs.set_xlabel('time (s)', fontsize=12)
axs.set_title('u Time Series (z>0)')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveSuffix = '_abovesurface'
saveName = 'u_ts' + saveSuffix + '.png'
plt.savefig(saveDir + saveName)
plt.show()




### plot the maximum particle velocity u along depth
xind = 25
yind = 5

z = []
zlen = []
for i in range(jobnum):
    z.append(data[i]['z'])
    zlen.append(len(z[i]))

t = []
tlen = []
for i in range(jobnum):
    t.append(data[i]['time'])
    tlen.append(len(t[i]))


umax = []
for i in range(jobnum):
    tmpx = t[i]
    tmp = []
    for zind in range(zlen[i]):
        tmpy = data[i]['u'][:,zind,yind,xind]
        tmpu = wvc.count_wave(tmpx,tmpy)[1]
        tmp.append(np.mean(tmpu)/2)
    umax.append(tmp)

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
markers = ['o', 's', 'd', '^']
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(z[i], umax[i], linewidth=None, color=colors[i], linestyle='', marker=markers[i], label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel(r'$u_{max}$ (m/s)', fontsize=12)
axs.grid()
axs.set_xlabel('z (m)', fontsize=12)
axs.set_title('Surface Elevation Time Series')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveName = 'umax_z' + '.png'
plt.savefig(saveDir + saveName)
plt.show()


### plot the maximum particle velocity w along depth
xind = 25
yind = 5

z = []
zlen = []
for i in range(jobnum):
    z.append(data[i]['z'])
    zlen.append(len(z[i]))

t = []
tlen = []
for i in range(jobnum):
    t.append(data[i]['time'])
    tlen.append(len(t[i]))


wmax = []
for i in range(jobnum):
    tmpx = t[i]
    tmp = []
    for zind in range(zlen[i]):
        tmpy = data[i]['w'][:,zind,yind,xind]
        tmpw = wvc.count_wave(tmpx,tmpy)[1]
        tmp.append(np.mean(tmpw)/2)
    wmax.append(tmp)

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
markers = ['o', 's', 'd', '^']
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(z[i], wmax[i], linewidth=None, color=colors[i], linestyle='', marker=markers[i], label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel(r'$w_{max}$ (m/s)', fontsize=12)
axs.grid()
axs.set_xlabel('z (m)', fontsize=12)
axs.set_title('Surface Elevation Time Series')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveName = 'wmax_z' + '.png'
plt.savefig(saveDir + saveName)
plt.show()


### plot the maximum pressure fluctuation along depth
xind = 25
yind = 5

k = 1
d = 30
z = -5
np.cosh(k*(d+z)) / np.cosh(k*d)

z = []
zlen = []
for i in range(jobnum):
    z.append(data[i]['z'])
    zlen.append(len(z[i]))

t = []
tlen = []
for i in range(jobnum):
    t.append(data[i]['time'])
    tlen.append(len(t[i]))


pmax = []
for i in range(jobnum):
    tmpx = t[i]
    tmp = []
    for zind in range(zlen[i]):
        tmpy = data[i]['p'][:,zind,yind,xind] # + rho*g*z[i][zind]
        tmpy = tmpy - tmpy.mean()
        tmpp = wvc.count_wave(tmpx,tmpy)[1]
        tmp.append(np.mean(tmpp)/2)
    pmax.append(tmp)

fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)

colors = plt.cm.jet(np.linspace(0,1,jobnum))
markers = ['o', 's', 'd', '^']
# colors = ['k']
# linestyles = ['-']
labels = jobname

for i in range(jobnum):
    axs.plot(z[i], pmax[i], linewidth=None, color=colors[i], linestyle='', marker=markers[i], label=labels[i])
# axs.set_ylim(ylims[0])
axs.set_ylabel(r'$p\prime_{max}$ ($N / m^2$)', fontsize=12)
axs.grid()
axs.set_xlabel('z (m)', fontsize=12)
axs.set_title('Surface Elevation Time Series')
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + prjname + '/'
saveName = 'pflcmax_z' + '.png'
plt.savefig(saveDir + saveName)
plt.show()
