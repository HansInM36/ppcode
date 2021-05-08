import os
import sys
sys.path.append('/scratch/ppcode')
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
from scipy.optimize import curve_fit
import funcs
import matplotlib.pyplot as plt

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=int(days)) \
           + datetime.timedelta(hours=int(hours)) \
           + datetime.timedelta(minutes=int(minutes)) \
           + datetime.timedelta(seconds=round(seconds)) \
           - datetime.timedelta(days=366)

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

""" WRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "WRFOUT_NODA_20150701.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")


dimlist = list(data.dimensions)
varlist = list(data.variables)

time_org = data.variables['time'][:]
dateTime = []
time = []
for tInd in range(time_org.size):
    dateTime.append(datenum_to_datetime(time_org[tInd]))
    time.append(dateTime[tInd].timestamp() - dateTime[0].timestamp())

Z = data.variables['Z'][:]
U = data.variables['U'][:]
V = data.variables['V'][:]
THETA = data.variables['THETA'][:] # it seems that THETA is not potential temperature

U10 = data.variables['U10'][:]
V10 = data.variables['V10'][:]

UV = np.sqrt(np.power(U,2) + np.power(V,2))
WD = funcs.wd(U,V)
    


""" PALM data """
jobName = 'WRFPALM_20150701'
dir = '/scratch/palmdata/JOBS/' + jobName

run_no_list = ['.000','.001','.002','.003','.004','.005','.006','.007']
#run_no_list = ['.000','.001','.002','.003']

uList = []
vList = []

for run_no in run_no_list:
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    tSeq = np.array(input_data.variables['time'][:], dtype=type(input_data.variables['time']))
    tNum = tSeq.size

    for tInd in range(tNum):
        t, zSeq, ySeq, xuSeq, uSeq = palm_3d_single(dir, jobName, run_no, tInd, 'u')
        t, zSeq, ySeq, xuSeq, vSeq = palm_3d_single(dir, jobName, run_no, tInd, 'v')
        
        u10Seq = (uSeq[1,:,:] + uSeq[2,:,:])/2
        v10Seq = (vSeq[1,:,:] + vSeq[2,:,:])/2
        
        u = u10Seq[:,0].mean(), u10Seq[:,-1].mean(), u10Seq[0,:].mean(), u10Seq[-1,:].mean(), u10Seq.mean() 
        v = v10Seq[:,0].mean(), v10Seq[:,-1].mean(), v10Seq[0,:].mean(), v10Seq[-1,:].mean(), v10Seq.mean() 
        
        uList.append(u)
        vList.append(v)



""" PALM dynamic driver data """

readDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/INPUT'
readName = "WRFPALM_20150701_dynamic"

nx, ny, nz = 384, 384, 96
dx, dy, dz = 40, 40, 10

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)

tSeq = data.variables['time'][:]

zSeq = data.variables['z'][:]
zwSeq = data.variables['zw'][:]

left_u = np.array(data.variables['ls_forcing_left_u'])
right_u = np.array(data.variables['ls_forcing_right_u'])
south_u = np.array(data.variables['ls_forcing_south_u'])
north_u = np.array(data.variables['ls_forcing_north_u'])

left_v = np.array(data.variables['ls_forcing_left_v'])
right_v = np.array(data.variables['ls_forcing_right_v'])
south_v = np.array(data.variables['ls_forcing_south_v'])
north_v = np.array(data.variables['ls_forcing_north_v'])

data.close()

u_List = []
v_List = []

for tInd in range(1,13):
    ul = (left_u[tInd,0].mean() + left_u[tInd,1].mean()) / 2
    vl = (left_v[tInd,0].mean() + left_v[tInd,1].mean()) / 2
    
    ur = (right_u[tInd,0].mean() + right_u[tInd,1].mean()) / 2
    vr = (right_v[tInd,0].mean() + right_v[tInd,1].mean()) / 2
    
    us = (south_u[tInd,0].mean() + south_u[tInd,1].mean()) / 2
    vs = (south_v[tInd,0].mean() + south_v[tInd,1].mean()) / 2
    
    un = (north_u[tInd,0].mean() + north_u[tInd,1].mean()) / 2
    vn = (north_v[tInd,0].mean() + north_v[tInd,1].mean()) / 2
    
    u_List.append((ul,ur,us,un))
    v_List.append((vl,vr,vs,vn))


""" time series of u10, v10 """
fig, ax = plt.subplots(figsize=(12.0,4.0))
plt.plot(tSeq[1:13]/3600, [uList[i][0] for i in range(12)], label='PALM left', marker='', linestyle='-', linewidth=1.0, color='r')
plt.plot(tSeq[1:13]/3600, [uList[i][1] for i in range(12)], label='PALM right', marker='', linestyle='-', linewidth=1.0, color='b')
plt.plot(tSeq[1:13]/3600, [uList[i][2] for i in range(12)], label='PALM south', marker='', linestyle='-', linewidth=1.0, color='g')
plt.plot(tSeq[1:13]/3600, [uList[i][3] for i in range(12)], label='PALM north ', marker='', linestyle='-', linewidth=1.0, color='y')
plt.plot(tSeq[1:13]/3600, [u_List[i][0] for i in range(12)], label='driver left', marker='<', linestyle='', linewidth=1.0, color='r')
plt.plot(tSeq[1:13]/3600, [u_List[i][1] for i in range(12)], label='driver right', marker='>', linestyle='', linewidth=1.0, color='b')
plt.plot(tSeq[1:13]/3600, [u_List[i][2] for i in range(12)], label='driver south', marker='v', linestyle='', linewidth=1.0, color='g')
plt.plot(tSeq[1:13]/3600, [u_List[i][3] for i in range(12)], label='driver north ', marker='^', linestyle='', linewidth=1.0, color='y')
plt.xlabel('t (h)', fontsize=12)
plt.ylabel('u (m/s)', fontsize=12)
xaxis_min = 1.0
xaxis_max = 12.0
xaxis_d = 1.0
yaxis_min = -10.0
yaxis_max = -4.0
yaxis_d = 1.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
#saveName = 'pr_ti_t' + str(np.round(t,1)) + '.png'
#saveDir = '/scratch/projects/EERA-SP3/photo/ti_pr'
#if not os.path.exists(saveDir):
#    os.makedirs(saveDir)
#plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()  


""" time series of u10, v10 """
fig, ax = plt.subplots(figsize=(12.0,4.0))
plt.plot(tSeq[1:13]/3600, [uList[i][4] for i in range(12)], label='PALM mean', linestyle='-', linewidth=1.0, color='b')
plt.plot(tSeq[1:13]/3600, U10[1:13], label='WRF', linestyle='--', linewidth=1.0, color='r')
plt.plot(tSeq[1:13]/3600, [uList[i][0] for i in range(12)], label='PALM left', marker='<', linestyle='-', linewidth=1.0, color='k')
plt.plot(tSeq[1:13]/3600, [uList[i][1] for i in range(12)], label='PALM right', marker='>', linestyle='-', linewidth=1.0, color='k')
plt.plot(tSeq[1:13]/3600, [uList[i][2] for i in range(12)], label='PALM south', marker='v', linestyle='-', linewidth=1.0, color='k')
plt.plot(tSeq[1:13]/3600, [uList[i][3] for i in range(12)], label='PALM north ', marker='^', linestyle='-', linewidth=1.0, color='k')
plt.xlabel('t (h)', fontsize=12)
plt.ylabel('u (m/s)', fontsize=12)
xaxis_min = 1.0
xaxis_max = 12.0
xaxis_d = 1.0
yaxis_min = -10.0
yaxis_max = -4.0
yaxis_d = 1.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'u10_cmp_WRF_PALM.png'
saveDir = '/scratch/projects/EERA-SP3/photo/uv10_cmp'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()   