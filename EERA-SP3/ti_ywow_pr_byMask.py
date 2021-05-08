import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/palm')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import sliceDataClass as sdc
import funcs
import palm_funcs
import matplotlib.pyplot as plt

def find_nearest_smaller_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be smaller or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] <= v:
        return ind
    else:
        return ind-1

def find_nearest_larger_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be larger or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] > v:
        return ind
    else:
        return ind+1

    
""" estimate the TI with masked data """
""" EERASP3_2_yw """
jobDir = '/scratch/palmdata/JOBS/EERASP3_2_yw'
jobName  = 'EERASP3_2_yw'
tSeq, zSeq, ySeq, xuSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'u')
tSeq, zSeq, yvSeq, xSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'v')
tSeq, zwSeq, ySeq, xSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w')

t_plot = np.arange(1,12) * 3600 # times of profiles that are going to be plotted

TIu_yw = np.zeros((t_plot.size, zSeq.size-1))
TIv_yw = np.zeros((t_plot.size, zSeq.size-1))
TIw_yw = np.zeros((t_plot.size, zSeq.size-1))


for zInd in range(1,zSeq.size): # exclude the zero height

    u_org = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'u', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    v_org = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'v', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org_up = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org_down = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd-1,zInd))

    # coordinates for scalar positions (exlude the grids at far right and far north where the interpolation is not available due to lack of data)
    z_seq = u_org[1]
    y_seq = u_org[2][:-1] # exclude the far north
    x_seq = v_org[3][:-1] # exclude the far east
    
    # interpolate vertically for w
    if zInd == 1:
        w_seq = 0.5*(w_org_up[4] + 0)
    else:
        w_seq = 0.5*(w_org_up[4] + w_org_down[4])
    w_seq = w_seq[:,:,:-1,:-1]
    
    # interpolate horizontally for u
    u_seq = 0.5*(u_org[4][:,:,:,:-1] + u_org[4][:,:,:,1:])
    u_seq = u_seq[:,:,:-1,:]
    
    # interpolate horizontally for v
    v_seq = 0.5*(v_org[4][:,:,:-1,:] + v_org[4][:,:,1:,:])
    v_seq = v_seq[:,:,:,:-1]

    # interpolate all data into the scalar position in the C-grid
    for i in range(t_plot.size):
        print('time: ' + str(t_plot[i]/3600) + 'h, ' + 'height: ' + str(z_seq[0]) + 'm')
        tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
        tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])

        t = tSeq[tInd_0:tInd_1+1]
        u = u_seq[tInd_0:tInd_1+1]
        v = v_seq[tInd_0:tInd_1+1]
        w = w_seq[tInd_0:tInd_1+1]
        
        u_ave = np.average(u, axis=0)
        v_ave = np.average(v, axis=0)
        uv_ave = np.power(np.power(u_ave,2) + np.power(v_ave,2), 0.5)
        
        sgm2u = np.zeros(u_ave.shape)
        sgm2v = np.zeros(u_ave.shape)
        sgm2w = np.zeros(u_ave.shape)
        
        # calculate the 3d TI at all grid points
        for yInd in range(y_seq.size):
            for xInd in range(x_seq.size):
                u_ = u[:,0,yInd,xInd] - u_ave[0,yInd,xInd]
                v_ = v[:,0,yInd,xInd] - v_ave[0,yInd,xInd]
                w_ = w[:,0,yInd,xInd] - 0
                # detrend
                u_ = funcs.detrend(t, u_)
                v_ = funcs.detrend(t, v_)
                w_ = funcs.detrend(t, w_)
                # variance
                sgm2u[0,yInd,xInd] = np.var(u_)
                sgm2v[0,yInd,xInd] = np.var(v_)
                sgm2w[0,yInd,xInd] = np.var(w_)
        
        TIu_yw[i,zInd-1] = np.mean(np.power(sgm2u,0.5)/uv_ave)
        TIv_yw[i,zInd-1] = np.mean(np.power(sgm2v,0.5)/uv_ave)
        TIw_yw[i,zInd-1] = np.mean(np.power(sgm2w,0.5)/uv_ave)



""" EERASP3_2_ow """
jobDir = '/scratch/palmdata/JOBS/EERASP3_2_ow'
jobName  = 'EERASP3_2_ow'
tSeq, zSeq, ySeq, xuSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'u')
tSeq, zSeq, yvSeq, xSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'v')
tSeq, zwSeq, ySeq, xSeq = palm_funcs.getInfo_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w')

t_plot = np.arange(1,12) * 3600 # times of profiles that are going to be plotted

TIu_ow = np.zeros((t_plot.size, zSeq.size-1))
TIv_ow = np.zeros((t_plot.size, zSeq.size-1))
TIw_ow = np.zeros((t_plot.size, zSeq.size-1))

for zInd in range(1,zSeq.size): # exclude the zero height

    u_org = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'u', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    v_org = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'v', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org_up = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd,zInd+1))
    w_org_down = palm_funcs.getData_palm_mask(jobDir, jobName, 'M01', ['.000','.001','.002'], 'w', [0,86400], (0,xSeq.size), (0,ySeq.size), (zInd-1,zInd))

    # coordinates for scalar positions (exlude the grids at far right and far north where the interpolation is not available due to lack of data)
    z_seq = u_org[1]
    y_seq = u_org[2][:-1] # exclude the far north
    x_seq = v_org[3][:-1] # exclude the far east
    
    # interpolate vertically for w
    if zInd == 1:
        w_seq = 0.5*(w_org_up[4] + 0)
    else:
        w_seq = 0.5*(w_org_up[4] + w_org_down[4])
    w_seq = w_seq[:,:,:-1,:-1]
    
    # interpolate horizontally for u
    u_seq = 0.5*(u_org[4][:,:,:,:-1] + u_org[4][:,:,:,1:])
    u_seq = u_seq[:,:,:-1,:]
    
    # interpolate horizontally for v
    v_seq = 0.5*(v_org[4][:,:,:-1,:] + v_org[4][:,:,1:,:])
    v_seq = v_seq[:,:,:,:-1]

    # interpolate all data into the scalar position in the C-grid
    for i in range(t_plot.size):
        print('time: ' + str(t_plot[i]/3600) + 'h, ' + 'height: ' + str(z_seq[0]) + 'm')
        tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
        tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])

        t = tSeq[tInd_0:tInd_1+1]
        u = u_seq[tInd_0:tInd_1+1]
        v = v_seq[tInd_0:tInd_1+1]
        w = w_seq[tInd_0:tInd_1+1]
        
        u_ave = np.average(u, axis=0)
        v_ave = np.average(v, axis=0)
        uv_ave = np.power(np.power(u_ave,2) + np.power(v_ave,2), 0.5)
        
        sgm2u = np.zeros(u_ave.shape)
        sgm2v = np.zeros(u_ave.shape)
        sgm2w = np.zeros(u_ave.shape)
        
        # calculate the 3d TI at all grid points
        for yInd in range(y_seq.size):
            for xInd in range(x_seq.size):
                u_ = u[:,0,yInd,xInd] - u_ave[0,yInd,xInd]
                v_ = v[:,0,yInd,xInd] - v_ave[0,yInd,xInd]
                w_ = w[:,0,yInd,xInd] - 0
                # detrend
                u_ = funcs.detrend(t, u_)
                v_ = funcs.detrend(t, v_)
                w_ = funcs.detrend(t, w_)
                # variance
                sgm2u[0,yInd,xInd] = np.var(u_)
                sgm2v[0,yInd,xInd] = np.var(v_)
                sgm2w[0,yInd,xInd] = np.var(w_)
        
        TIu_ow[i,zInd-1] = np.mean(np.power(sgm2u,0.5)/uv_ave)
        TIv_ow[i,zInd-1] = np.mean(np.power(sgm2v,0.5)/uv_ave)
        TIw_ow[i,zInd-1] = np.mean(np.power(sgm2w,0.5)/uv_ave)
        


""" TI cmp """
for i in range(t_plot.size):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    plt.plot(TIu_yw[i]*100, zSeq[1:], label='TIu-yw', marker='', linestyle='-', linewidth=1.0, color='r')
    plt.plot(TIv_yw[i]*100, zSeq[1:], label='TIv-yw', marker='', linestyle='-', linewidth=1.0, color='b')
    plt.plot(TIw_yw[i]*100, zSeq[1:], label='TIw-yw', marker='', linestyle='-', linewidth=1.0, color='g')
    plt.plot(TIu_ow[i]*100, zSeq[1:], label='TIu-ow', marker='', linestyle='--', linewidth=1.0, color='r')
    plt.plot(TIv_ow[i]*100, zSeq[1:], label='TIv-ow', marker='', linestyle='--', linewidth=1.0, color='b')
    plt.plot(TIw_ow[i]*100, zSeq[1:], label='TIw-ow', marker='', linestyle='--', linewidth=1.0, color='g')
    plt.xlabel(r"$\mathrm{TI}$ (%)")
    plt.ylabel('z (m)')
    xaxis_min = 0
    xaxis_max = 3
    xaxis_d = 0.5
    yaxis_min = 0
    yaxis_max = 200.0
    yaxis_d = 20.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    plt.legend(bbox_to_anchor=(0.64,0.72), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t =' + str(np.round(t_plot[i]/3600,1)) + 'h') 
    fig.tight_layout() # adjust the layout
    saveName = 'TI_pr_t' + str(np.round(t_plot[i]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/TI_pr/1st'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()