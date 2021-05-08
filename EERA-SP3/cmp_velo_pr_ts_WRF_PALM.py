import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

def palm_3d_single(dir, jobName, run_no, tInd, var):
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    t = np.array(input_data.variables['time'][tInd], dtype=float)
    varSeq = np.array(input_data.variables[var][tInd], dtype=float)
    
    # extract the values of all dimensions of the var
    zName = list(input_data.variables[var].dimensions)[1] # the height name string
    zSeq = np.array(input_data.variables[zName][:], dtype=float) # array of height levels
    yName = list(input_data.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_data.variables[yName][:], dtype=float) # array of height levels
    xName = list(input_data.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_data.variables[xName][:], dtype=float) # array of height levels
    return t, zSeq, ySeq, xSeq, varSeq


""" WRF """
readDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/INPUT'
readName = "WRFPALM_20150701_dynamic"

nx, ny, nz = 256, 256, 96
dx, dy, dz = 10, 10, 10

wrf = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(wrf.dimensions)
varlist = list(wrf.variables)

TIME = np.array(wrf.variables['time'][:], dtype=float)
Y = np.array(wrf.variables['y'][:], dtype=float)
Z = np.array(wrf.variables['z'][:], dtype=float)
left_U = np.array(wrf.variables['ls_forcing_left_u'][:], dtype=float)
left_V = np.array(wrf.variables['ls_forcing_left_v'][:], dtype=float)
right_U = np.array(wrf.variables['ls_forcing_right_u'][:], dtype=float)
right_V = np.array(wrf.variables['ls_forcing_right_v'][:], dtype=float)
south_U = np.array(wrf.variables['ls_forcing_south_u'][:], dtype=float)
south_V = np.array(wrf.variables['ls_forcing_south_v'][:], dtype=float)
north_U = np.array(wrf.variables['ls_forcing_north_u'][:], dtype=float)
north_V = np.array(wrf.variables['ls_forcing_north_v'][:], dtype=float)


""" PALM """
jobName  = 'WRFPALM_20150701'
dir = "/scratch/palmdata/JOBS/" + jobName
run_no_list = ['.000','.001','.002','.003','.004','.005','.006','.007']

time = []
u10 = []
v10 = []
u_av = []
v_av = []
left_u = []
left_v = []
right_u = []
right_v = []
south_u = []
south_v = []
north_u = []
north_v = []

for i in range(len(run_no_list)):
    run_no = run_no_list[i]
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    time_ = np.array(input_data.variables['time'][:], dtype=type(input_data.variables['time']))
    tNum_ = time_.size
    
    for tInd in range(tNum_):
        t, zSeq, ySeq, xSeq, uSeq = palm_3d_single(dir, jobName, run_no, tInd, 'u')
        t, zSeq, ySeq, xSeq, vSeq = palm_3d_single(dir, jobName, run_no, tInd, 'v')
        
        time.append(t)
        u_av.append(np.average(uSeq,axis=(1,2)))
        v_av.append(np.average(vSeq,axis=(1,2)))
        left_u.append(np.average(uSeq[:,:,0],1))
        left_v.append(np.average(vSeq[:,:,0],1))
        right_u.append(np.average(uSeq[:,:,-1],1))
        right_v.append(np.average(vSeq[:,:,-1],1))
        south_u.append(np.average(uSeq[:,0,:],1))
        south_v.append(np.average(vSeq[:,0,:],1))
        north_u.append(np.average(uSeq[:,-1,:],1))
        north_v.append(np.average(vSeq[:,-1,:],1))
        
        fu = interp1d(zSeq, uSeq, axis=0)
        u10.append(fu(10))
        fv = interp1d(zSeq, vSeq, axis=0)
        v10.append(fv(10))

time = np.array(time)
u_av = np.array(u_av)
v_av = np.average(v_av)
left_u = np.array(left_u)
left_v = np.array(left_v)
right_u = np.array(right_u)
right_v = np.array(right_v)
south_u = np.array(south_u)
south_v = np.array(south_v)
north_u = np.array(north_u)
north_v = np.array(north_v)
u10 = np.array(u10)
v10 = np.array(v10)



""" u left boundary plane """
fig, axs = plt.subplots(3,1, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (-14.0, -2.0, 2.0)
cbreso = 32 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
# WRF
CS = axs[0].contourf(TIME[1:]/3600, Z, np.transpose(np.average(left_U[1:],axis=2)), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[0].set_xlim([1,12])
axs[0].set_ylim([0,1200])
axs[0].set_ylabel('z (m)', fontsize=12)
#axs[0].set_xlabel('t (h)', fontsize=12)
axs[0].text(1.02, 0.16, 'Driver - left', transform=axs[0].transAxes, rotation=90, fontsize=12)
# PALM - left
CS = axs[1].contourf(time/3600, zSeq, np.transpose(left_u), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[1].set_xlim([1,12])
axs[1].set_ylim([0,1200])
axs[1].set_ylabel('z (m)', fontsize=12)
#axs[1].set_xlabel('t (h)', fontsize=12)
axs[1].text(1.02, 0.2, 'PALM - left', transform=axs[1].transAxes, rotation=90, fontsize=12)
# PALM - horizon
CS = axs[2].contourf(time/3600, zSeq, np.transpose(u_av), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[2].set_xlim([1,12])
axs[2].set_ylim([0,1200])
axs[2].set_ylabel('z (m)', fontsize=12)
axs[2].set_xlabel('t (h)', fontsize=12)
axs[2].text(1.02, 0.1, 'PALM - mean', transform=axs[2].transAxes, rotation=90, fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)

axs[0].set_title('u left boundary plane')

saveName = 'left_velo_pr_ts.png'
saveDir = '/scratch/projects/EERA-SP3/photo/cmp_velo_pr_ts'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()



""" u right boundary plane """
fig, axs = plt.subplots(3,1, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (-14.0, -2.0, 2.0)
cbreso = 32 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
# WRF
CS = axs[0].contourf(TIME[1:]/3600, Z, np.transpose(np.average(right_U[1:],axis=2)), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[0].set_xlim([1,12])
axs[0].set_ylim([0,1200])
axs[0].set_ylabel('z (m)', fontsize=12)
#axs[0].set_xlabel('t (h)', fontsize=12)
axs[0].text(1.02, 0.16, 'Driver - right', transform=axs[0].transAxes, rotation=90, fontsize=12)
# PALM - right
CS = axs[1].contourf(time/3600, zSeq, np.transpose(right_u), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[1].set_xlim([1,12])
axs[1].set_ylim([0,1200])
axs[1].set_ylabel('z (m)', fontsize=12)
#axs[1].set_xlabel('t (h)', fontsize=12)
axs[1].text(1.02, 0.2, 'PALM - right', transform=axs[1].transAxes, rotation=90, fontsize=12)
# PALM - horizon
CS = axs[2].contourf(time/3600, zSeq, np.transpose(u_av), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[2].set_xlim([1,12])
axs[2].set_ylim([0,1200])
axs[2].set_ylabel('z (m)', fontsize=12)
axs[2].set_xlabel('t (h)', fontsize=12)
axs[2].text(1.02, 0.1, 'PALM - mean', transform=axs[2].transAxes, rotation=90, fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)

axs[0].set_title('u right boundary plane')

saveName = 'right_velo_pr_ts.png'
saveDir = '/scratch/projects/EERA-SP3/photo/cmp_velo_pr_ts'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" u south boundary plane """
fig, axs = plt.subplots(3,1, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (-14.0, -2.0, 2.0)
cbreso = 32 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
# WRF
CS = axs[0].contourf(TIME[1:]/3600, Z, np.transpose(np.average(south_U[1:],axis=2)), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[0].set_xlim([1,12])
axs[0].set_ylim([0,1200])
axs[0].set_ylabel('z (m)', fontsize=12)
#axs[0].set_xlabel('t (h)', fontsize=12)
axs[0].text(1.02, 0.16, 'Driver - south', transform=axs[0].transAxes, rotation=90, fontsize=12)
# PALM - south
CS = axs[1].contourf(time/3600, zSeq, np.transpose(south_u), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[1].set_xlim([1,12])
axs[1].set_ylim([0,1200])
axs[1].set_ylabel('z (m)', fontsize=12)
#axs[1].set_xlabel('t (h)', fontsize=12)
axs[1].text(1.02, 0.2, 'PALM - south', transform=axs[1].transAxes, rotation=90, fontsize=12)
# PALM - horizon
CS = axs[2].contourf(time/3600, zSeq, np.transpose(u_av), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[2].set_xlim([1,12])
axs[2].set_ylim([0,1200])
axs[2].set_ylabel('z (m)', fontsize=12)
axs[2].set_xlabel('t (h)', fontsize=12)
axs[2].text(1.02, 0.1, 'PALM - mean', transform=axs[2].transAxes, rotation=90, fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)

axs[0].set_title('u south boundary plane')

saveName = 'south_velo_pr_ts.png'
saveDir = '/scratch/projects/EERA-SP3/photo/cmp_velo_pr_ts'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" u north boundary plane """
fig, axs = plt.subplots(3,1, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (-14.0, -2.0, 2.0)
cbreso = 32 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
# WRF
CS = axs[0].contourf(TIME[1:]/3600, Z, np.transpose(np.average(north_U[1:],axis=2)), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[0].set_xlim([1,12])
axs[0].set_ylim([0,1200])
axs[0].set_ylabel('z (m)', fontsize=12)
#axs[0].set_xlabel('t (h)', fontsize=12)
axs[0].text(1.02, 0.16, 'Driver - north', transform=axs[0].transAxes, rotation=90, fontsize=12)
# PALM - north
CS = axs[1].contourf(time/3600, zSeq, np.transpose(north_u), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[1].set_xlim([1,12])
axs[1].set_ylim([0,1200])
axs[1].set_ylabel('z (m)', fontsize=12)
#axs[1].set_xlabel('t (h)', fontsize=12)
axs[1].text(1.02, 0.2, 'PALM - north', transform=axs[1].transAxes, rotation=90, fontsize=12)
# PALM - horizon
CS = axs[2].contourf(time/3600, zSeq, np.transpose(u_av), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs[2].set_xlim([1,12])
axs[2].set_ylim([0,1200])
axs[2].set_ylabel('z (m)', fontsize=12)
axs[2].set_xlabel('t (h)', fontsize=12)
axs[2].text(1.02, 0.1, 'PALM - mean', transform=axs[2].transAxes, rotation=90, fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)

axs[0].set_title('u north boundary plane')

saveName = 'north_velo_pr_ts.png'
saveDir = '/scratch/projects/EERA-SP3/photo/cmp_velo_pr_ts'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" u10 contour """
for tInd in range(time.size):
    fig, axs = plt.subplots(1,1, constrained_layout=False)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    
    ### contour
    vMin, vMax, vDelta = (-10.0, -4.0, 1.0)
    cbreso = 32 # resolution of colorbar
    levels = np.linspace(vMin, vMax, cbreso + 1)
    # WRF
    CS = axs.contourf(xSeq, ySeq, u10[tInd], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    axs.set_xlim([0,15360])
    axs.set_ylim([0,15360])
    axs.set_ylabel('y (m)', fontsize=12)
    axs.set_xlabel('x (m)', fontsize=12)
    
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    cbar = plt.colorbar(CS, ax=axs.ravel().tolist(), orientation='vertical', ticks=cbartickList, fraction=.1)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)
    
    axs.set_title('u10')
    
    saveName = 'u10_t' + str(np.round(time[tInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/velo10'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()