import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt


def get_wave(dir, file):
    data = Dataset(dir + '/' + file, "r", format="NETCDF4")

    tSeq = np.array(data.variables['time'][:]).astype(float)
    tNum = tSeq.size
    ySeq = np.array(data.variables['y'][:]).astype(float)
    yNum = ySeq.size
    xSeq = np.array(data.variables['x'][:]).astype(float)
    xNum = xSeq.size

    etaSeq = np.array(data.variables['eta'][:,:,:]).astype(float)
    phiSeq = np.array(data.variables['phi'][:,:,:]).astype(float)

    uSeq = np.zeros(phiSeq.shape)
    vSeq = np.zeros(phiSeq.shape)
    wSeq = np.zeros(phiSeq.shape)

    cosxSeq = np.zeros(phiSeq.shape)
    sinxSeq = np.zeros(phiSeq.shape)
    cosySeq = np.zeros(phiSeq.shape)
    sinySeq = np.zeros(phiSeq.shape)

    for i in range(xNum):
        if i == 0:
            uSeq[:,:,i] = (phiSeq[:,:,i+1] - phiSeq[:,:,i]) / (xSeq[i+1] - xSeq[i])
            deta = etaSeq[:,:,i+1] - etaSeq[:,:,i]
            dx = xSeq[i+1] - xSeq[i]
        elif i == xNum-1:
            uSeq[:,:,i] = (phiSeq[:,:,i] - phiSeq[:,:,i-1]) / (xSeq[i] - xSeq[i-1])
            deta = etaSeq[:,:,i] - etaSeq[:,:,i-1]
            dx = xSeq[i] - xSeq[i-1]
        else:
            uSeq[:,:,i] = (phiSeq[:,:,i+1] - phiSeq[:,:,i-1]) / (xSeq[i+1] - xSeq[i-1])
            deta = etaSeq[:,:,i+1] - etaSeq[:,:,i-1]
            dx = xSeq[i+1] - xSeq[i-1]
        cosxSeq[:,:,i] = dx / np.sqrt(dx**2 + deta**2)
        sinxSeq[:,:,i] = deta / np.sqrt(dx**2 + deta**2)

    for j in range(yNum):
        if j == 0:
            vSeq[:,j,:] = (phiSeq[:,j+1,:] - phiSeq[:,j,:]) / (ySeq[j+1] - ySeq[j])
            deta = etaSeq[:,j+1,:] - etaSeq[:,j,:]
            dy = ySeq[j+1] - ySeq[j]
        elif j == yNum-1:
            vSeq[:,j,:] = (phiSeq[:,j,:] - phiSeq[:,j-1,:]) / (ySeq[j] - ySeq[j-1])
            deta = etaSeq[:,j,:] - etaSeq[:,j-1,:]
            dy = ySeq[j] - ySeq[j-1]
        else:
            vSeq[:,j,:] = (phiSeq[:,j+1,:] - phiSeq[:,j-1,:]) / (ySeq[j+1] - ySeq[j-1])
            deta = etaSeq[:,j+1,:] - etaSeq[:,j-1,:]
            dy = ySeq[j+1] - ySeq[j-1]
        cosySeq[:,j,:] = dy / np.sqrt(dy**2 + deta**2)
        sinySeq[:,j,:] = deta / np.sqrt(dy**2 + deta**2)

    for t in range(tNum):
        if t == 0:
            wSeq[t,:,:] = (etaSeq[t+1,:,:] - etaSeq[t,:,:]) / (tSeq[t+1] - tSeq[t])
        elif t == tNum-1:
            wSeq[t,:,:] = (etaSeq[t,:,:] - etaSeq[t-1,:,:]) / (tSeq[t] - tSeq[t-1])
        else:
            wSeq[t,:,:] = (etaSeq[t+1,:,:] - etaSeq[t-1,:,:]) / (tSeq[t+1] - tSeq[t-1])

    return tSeq, xSeq, ySeq, etaSeq, phiSeq, uSeq, vSeq, wSeq

dir = '/scratch/palmdata/pp/wwinta/data'
file = "waveData_regular.nc"
wave_regular = get_wave(dir, file)

dir = '/scratch/palmdata/pp/wwinta/data'
file = "waveData_irregular.nc"
wave_irregular = get_wave(dir, file)

tSeq, xSeq, ySeq, etaSeq, phiSeq, uSeq, vSeq, wSeq = wave_irregular

""" animation of eta field """
vMin, vMax, vDelta = (-1.6, 1.6, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,100,1):
    fig, axs = plt.subplots(figsize=(6.0,4.6), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = etaSeq[tInd]
    # v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{\eta}$" + ' (m)', fontsize=12)
    plt.xlim([0,1000])
    plt.ylim([0,600])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/palmdata/pp/wwinta/animation/eta_irregular'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')


""" animation of u field """
vMin, vMax, vDelta = (-1.5, 1.5, 0.5)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,100,1):
    fig, axs = plt.subplots(figsize=(6.0,4.6), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = uSeq[tInd]
    # v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='bwr', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,1000])
    plt.ylim([0,600])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/palmdata/pp/wwinta/animation/u_irregular'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')

""" animation of w field """
vMin, vMax, vDelta = (-4, 4, 1)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,100,1):
    fig, axs = plt.subplots(figsize=(6.0,4.6), constrained_layout=False)
    x_ = xSeq
    y_ = ySeq
    v_ = wSeq[tInd]
    # v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='bwr', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{w}$" + ' (m/s)', fontsize=12)
    plt.xlim([0,1800])
    plt.ylim([0,1800])
    plt.ylabel('y (m)', fontsize=12)
    plt.xlabel('x (m)', fontsize=12)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/HOSdata/pp/' + jobName + '/animation/w'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    # plt.show()
    plt.close('all')



""" time series of eta """
fig = plt.subplots(figsize=(5.2,3))
t0, t1 = wave_regular[0], wave_irregular[0]
eta0, eta1 = wave_regular[1][:,32,128], wave_irregular[1][:,32,128]
plt.plot(t0, eta0, label='regular', linewidth=1.0, linestyle='-', color='r')
plt.plot(t1, eta1, label='irregular', linewidth=1.0, linestyle='-', color='b')
plt.xlabel('t (s)')
plt.ylabel(r"$\eta$ (m)")
xaxis_min = 0
xaxis_max = 80
yaxis_min = -2
yaxis_max = 2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.66,0.8), loc=6, borderaxespad=0)
plt.grid()
saveName = 'eta' + '_ts' + '.png'
saveDir = prjDir + '/photo/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()


""" time series of u """
fig = plt.subplots(figsize=(5.2,3))
t0, t1 = wave_regular[0], wave_irregular[0]
u0, u1 = wave_regular[3][:,32,128], wave_irregular[3][:,32,128]
plt.plot(t0, u0, label='regular', linewidth=1.0, linestyle='-', color='r')
plt.plot(t1, u1, label='irregular', linewidth=1.0, linestyle='-', color='b')
plt.xlabel('t (s)')
plt.ylabel(r"u (m/s)")
xaxis_min = 0
xaxis_max = 80
yaxis_min = -2
yaxis_max = 2
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.66,0.8), loc=6, borderaxespad=0)
plt.grid()
saveName = 'u' + '_ts' + '.png'
saveDir = prjDir + '/photo/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()



""" PSD """
fs = 1
PSD_list = []
for i in range(xNum):
    for j in range(yNum):
        vSeq = etaSeq[:,j,i]
        f_seq, tmp = scipy.signal.csd(vSeq, vSeq, fs, nperseg=120, noverlap=None)
        PSD_list.append(tmp)
PSD_seq = np.average(np.array(PSD_list), axis=0)

# plot
fig, ax = plt.subplots(figsize=(5.2,3))
plt.loglog(f_seq, PSD_seq, label='', linewidth=1.0, linestyle='-', color='k')
plt.xlabel('f (1/s)')
plt.ylabel('S' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e3
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(0.02,0.42), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()
