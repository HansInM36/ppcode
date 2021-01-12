#!/usr/bin/python3.8

import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import funcs
import matplotlib.pyplot as plt

def velo_prb(dir, jobName, maskID, run_no_list, var):
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

def coh(p0_ind, p1_ind, t_para, tSeq, varSeq):
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    u0 = varSeq[:,p0_ind[2],p0_ind[1],p0_ind[0]]
    u1 = varSeq[:,p1_ind[2],p1_ind[1],p1_ind[0]]
    f0 = interp1d(tSeq, u0, kind='linear', fill_value='extrapolate')
    f1 = interp1d(tSeq, u1, kind='linear', fill_value='extrapolate')
    u0_ = f0(t_seq)
    u1_ = f1(t_seq)

    segNum = 120*fs
    freq, coh, co_coh, phase = funcs.coherence(u0_, u1_, fs, segNum)
    return t_seq, u0_, u1_, freq, coh, co_coh, phase


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind'
suffix = '_gs10'
ppDir = '/scratch/palmdata/pp/' + jobName + suffix

maskid = 'M04'

# cycle_no_list = ['.011','.012']
cycle_no_list = ['.022']
# cycle_no_list = ['.007','.008','.009','.010'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

var = 'u'
varName = 'coherence'
varUnit = ''
varName_save = 'uu_coherence'

tInd_start, tInd_end = 0, 75601
xInd_start, xInd_end = 0, -1
yInd_start, yInd_end = 0, 1
zInd_start, zInd_end = 0, -1

# read the output data of all cycle_no_list
nc_file_list = []
tSeq_list = []
varSeq_list = []
for i in range(cycle_num):
    input_file = prjDir + '/' + jobName + suffix + "/OUTPUT/" + jobName + suffix + "_masked_" + maskid + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    # varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd_start:tInd_end,zInd_start:zInd_end,yInd_start:yInd_end,xInd_start:xInd_end], dtype=type(nc_file_list[i].variables[var])))
    varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

# print(list(nc_file_list[0].dimensions)) #list all dimensions
# print(list(nc_file_list[0].variables)) #list all the variables
# print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

# extract the values of all dimensions of the var
zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
zSeq = zSeq.astype(float)
# zSeq = zSeq[zInd_start:zInd_end].astype(float)
zNum = zSeq.size
yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
ySeq = ySeq.astype(float)
# ySeq = ySeq[yInd_start:yInd_end].astype(float)
yNum = ySeq.size
# dy = ySeq[1] - ySeq[0]
xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
xSeq = xSeq.astype(float)
# xSeq = xSeq[xInd_start:xInd_end].astype(float)
# dx = xSeq[1] - xSeq[0]
xNum = xSeq.size

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tSeq = np.concatenate([tSeq_list[i] for i in range(cycle_num)], axis=0)
tSeq = tSeq.astype(float)
# tSeq = tSeq[tInd_start:tInd_end].astype(float)
varSeq = np.concatenate([varSeq_list[i] for i in range(cycle_num)], axis=0)
varSeq = varSeq.astype(float)

t_start = 108000.0
t_end = 110400.0
t_delta = 0.1
# t_start = 144000.0
# t_end = 146400.0
# t_delta = 0.1
fs = 1/t_delta
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)



segNum = 120*fs

xInds = (0, 2, 9, 9)
yInds = (0, 0, 0, 0)
zInds = (2, 2, 0, 0)

p0_coor = (xSeq[xInds[0]], ySeq[yInds[0]], zSeq[zInds[0]])
p1_coor = (xSeq[xInds[1]], ySeq[yInds[1]], zSeq[zInds[1]])
p2_coor = (xSeq[xInds[2]], ySeq[yInds[2]], zSeq[zInds[2]])
p3_coor = (xSeq[xInds[3]], ySeq[yInds[3]], zSeq[zInds[3]])
print(p0_coor, p1_coor, p2_coor, p3_coor)

dx = xSeq[xInds[1]] - xSeq[xInds[0]]
dy = ySeq[yInds[1]] - ySeq[yInds[0]]


u0_ = varSeq[:,zInds[0],yInds[0],xInds[0]]
u1_ = varSeq[:,zInds[1],yInds[1],xInds[1]]
u2_ = varSeq[:,zInds[2],yInds[2],xInds[2]]
u3_ = varSeq[:,zInds[3],yInds[3],xInds[3]]

# time interpolation
method_ = 'linear' # 'linear' or 'cubic'
f0 = interp1d(tSeq, u0_, kind=method_, fill_value='extrapolate')
f1 = interp1d(tSeq, u1_, kind=method_, fill_value='extrapolate')
f2 = interp1d(tSeq, u2_, kind=method_, fill_value='extrapolate')
f3 = interp1d(tSeq, u3_, kind=method_, fill_value='extrapolate')
u0 = f0(t_seq)
u1 = f1(t_seq)
u2 = f2(t_seq)
u3 = f3(t_seq)

# # white noise
# n0 = np.random.rand(t_num)
# n1 = np.random.rand(t_num)
# u0 += n0
# u1 += n1
# SNR0 = np.var(u0) / np.var(n0)
# SNR1 = np.var(u1) / np.var(n1)



""" group_plot_0 """
funcs.group_plot_0(t_seq - t_start, fs, u0-u0.mean(), u1-u1.mean())


# # check time series
fig, ax = plt.subplots(figsize=(8,4))
ind0, ind1 = 0, 70000
ax.plot(t_seq[ind0:ind1] - t_seq[0], u0[ind0:ind1], 'r-', label='p0')
ax.plot(t_seq[ind0:ind1] - t_seq[0], u1[ind0:ind1], 'b-', label='p1')
# plt.ylim(6, 10)
# plt.xlim(0, 120)
# ax.set_xlabel('t (s)', fontsize=12)
# ax.set_ylabel('u (m/s)', fontsize=12)
ax.text(0.56, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.grid()
plt.legend()
saveName = 'u_ts_120' + '_dx_' + str(np.round(dx,1)) + '_h_' + str(np.round(p0_coor[2])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()
plt.close()




segNum = 120*fs
# calculate coherence and phase
freq, coh01, co_coh01, phase01 = funcs.coherence(u0, u1, fs, segNum)
freq, coh02, co_coh02, phase02 = funcs.coherence(u0, u2, fs, segNum)
freq, coh03, co_coh03, phase03 = funcs.coherence(u0, u3, fs, segNum)

""" plot coherence and fitting curve """
def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)

f_out = 1
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

# dx = 40m
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], coh01[1:], linestyle='', marker='o', markersize=1, color='r', label='d = 40m')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh01[ind_in:ind_out], bounds=(0, [1, 100]))
ax.plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='r',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))
# dx = 80m
ax.plot(freq[1:], coh02[1:], linestyle='', marker='o', markersize=1, color='b', label='d = 80m')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh02[ind_in:ind_out], bounds=(0, [1, 100]))
ax.plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='b',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))
# dx = 120m
ax.plot(freq[1:], coh03[1:], linestyle='', marker='o', markersize=1, color='g', label='d = 120m')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh03[ind_in:ind_out], bounds=(0, [1, 100]))
ax.plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='g',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'fs = ' + str(fs) + 'Hz' + ', ' 'nperseg = ' + str(segNum), transform=ax.transAxes, fontsize=12)
# ax.text(0.56, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.5,0.75), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'uu_coh_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()



""" plot co-coherence """
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], co_coh01[1:], linestyle='', marker='o', markersize=1, color='r', label='d = 40m')
ax.plot(freq[1:], co_coh02[1:], linestyle='', marker='o', markersize=1, color='b', label='d = 80m')
ax.plot(freq[1:], co_coh03[1:], linestyle='', marker='o', markersize=1, color='g', label='d = 120m')

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('co-coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
# plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'uu_cocoh_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()



""" plot phase """
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], phase01[1:], linestyle='', marker='o', markersize=1, color='r', label='d = 40m')
ax.plot(freq[1:], phase02[1:], linestyle='', marker='o', markersize=1, color='b', label='d = 80m')
ax.plot(freq[1:], phase03[1:], linestyle='', marker='o', markersize=1, color='g', label='d = 120m')


plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('phase', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), \
['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'uu_phase_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()














""" plot coherence, co-coherence, phase in one figure """
f_out = 0.4
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

rNum, cNum = (1,3)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(4)

# coherence
axs[0].plot(freq[1:], coh[1:], linestyle='', marker='o', markersize=3, color='k')
# popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
# axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt))

axs[0].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
axs[0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[0].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0].grid()

axs[0].set_xlabel('f (1/s)', fontsize=12)
axs[0].set_ylabel('coherence', fontsize=12)

# axs[0].legend(bbox_to_anchor=(0.2,0.9), loc=6, borderaxespad=0, fontsize=10)

# co-coherence
axs[1].plot(freq[1:], co_coh[1:], linestyle='', marker='o', markersize=3, color='r')

axs[1].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
axs[1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[1].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1].grid()

axs[1].set_xlabel('f (1/s)', fontsize=12)
axs[1].set_ylabel('co-coherence', fontsize=12)

axs[1].set_title('dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', fontsize=12)

# phase
axs[2].plot(freq[1:], phase[1:], linestyle='', marker='o', markersize=3, color='b')

axs[2].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
axs[2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
# axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
          r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
axs[2].set_yticklabels(labels)
axs[2].grid()

axs[2].set_xlabel('f (1/s)', fontsize=12)
axs[2].set_ylabel('phase', fontsize=12)

fig.tight_layout()
saveName = 'coh_co-coh_phase' + '_dx_' + str(np.round(dx,1)) + '_h_' + str(np.round(p0_coor[2])) + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
