#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
import os
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm


def velo_pr_palm(dir, jobName, run_no_list, var):
    """ extract velocity data of specified probe groups """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
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
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, zSeq, varSeq

prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs10'
dir_1 = prjDir + '/' + jobName
tSeq_1, zSeq_1, varSeq_1 = velo_pr_palm(dir_1, jobName, ['.022'], 'u')



# time steps for plotting
ave_itv = 3600.0 # by default, the averaging interval is 3600s
                 # ind_back_num = np.floor(ave_itv / tDelta)
tplot_start = 3600.0*1 # must larger than ave_itv
tplot_end = 3600.0*1*5
tplot_delta = 3600.0*1*1
tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))
###

tSeq = tSeq_1
varSeq = varSeq_1
zSeq = zSeq_1
zNum = zSeq.size

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zind in range(zNum):
        f = interp1d(tSeq, varSeq[:,zind], kind='linear')
        varplot[zind] = f(tplot)
    varplotList.append(varplot)



fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=hubH, ls='--', c='black')
# plt.axhline(y=dampH, ls=':', c='black')
# plt.xlabel(varName_plot + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()

# interpolate the velocity at hub height
f = interp1d(zSeq, varplotList[-1], kind='linear')
varHub = f(hubH)




### plot non-dimensional u gradient profile
startH = 0.001
topH = 200.0
zNum_ = 21
uStar = 0.4
kappa = 0.4

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))
for i in range(tplotNum):
    zero = np.zeros(1)
    v_ = np.concatenate((zero, varplotList[i]))
    z_ = np.concatenate((zero, zSeq))
    f = interp1d(z_, v_, kind='linear', fill_value='extrapolate')
    z_ = np.linspace(startH,topH,zNum_)
    dz = (topH - startH) / (zNum_-1)
    v_ = funcs.calc_deriv_1st_FD(dz, f(z_))
    v_ = v_ * kappa * z_ / uStar
    plt.plot(v_, z_, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.xlabel(r"$\mathrm{\phi_m}$")
plt.ylabel('z (m)')
xaxis_min = -3
xaxis_max = 5
xaxis_d = 2
yaxis_min = 0
yaxis_max = 200.0
yaxis_d = 20.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'phi_m' + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()





### investigate the evolution of averaged velocities at various heights
# time steps for plotting
tplot_start = 3600.0*1 # must larger than ave_itv
tplot_end = 3600.0*1*40
tplot_delta = 3600.0*1
tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zind in range(zNum):
        f = interp1d(tSeq, varSeq[:,zind], kind='linear')
        varplot[zind] = f(tplot)
        # print(f(tplot))
    varplotList.append(varplot)


zList = [20.0, 40.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
zIndNum = len(zList)
uList = []
for z in zList:
    u_tmp = []
    for tInd in range(tplotNum):
        f = interp1d(zSeq, varplotList[tInd], kind='linear', fill_value='extrapolate')
        u_tmp.append(f(z))
    uList.append(u_tmp)


fig, ax = plt.subplots(figsize=(6,4))
colors = plt.cm.jet(np.linspace(0,1,zIndNum))
for i in range(zIndNum):
    np.array(uList[i]).mean()
    ax.plot(np.array(tplotList)/3600, uList[i], linewidth=1.0, linestyle='-', marker='', color=colors[i], label='H = ' + str(zList[i]) + 'm')
ax.set_xlabel('t (h)')
ax.set_ylabel(varName_plot + ' (m/s)')
xaxis_min = 0
xaxis_max = 40
xaxis_d = 4
yaxis_min = 5.0
yaxis_max = 10.0
yaxis_d = 0.5
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0)
plt.grid()
fig.tight_layout() # adjust the layout
saveName = 'velo_av_evo.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()
