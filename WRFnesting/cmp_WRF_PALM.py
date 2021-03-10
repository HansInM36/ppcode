import os
import sys
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
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

def pr_palm(dir, jobName, run_no_list, var):
    """ extract horizontal average of velocity at various times and heights """
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

    # dimensions = list(nc_file_list[0].dimensions)
    # vars = list(nc_file_list[0].variables)
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

""" WRF data """
readDir = '/scratch/projects/EERA-SP3'
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

UV = np.sqrt(np.power(U,2) + np.power(V,2))
WD = 270 - np.arctan(V / (U + 1e-6)) * 180/np.pi

""" PALM data """
jobName = 'WRFPALM_20150701'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq, zSeq, uSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'u')
tSeq, zSeq, vSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'v')
tSeq, zSeq, thetaSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'theta')

uvSeq = np.sqrt(np.power(uSeq,2) + np.power(vSeq,2))
wdSeq = 270 - np.arctan(vSeq / (uSeq + 1e-6)) * 180/np.pi


""" profile of horizontal velocity """
fig, ax = plt.subplots(figsize=(4.5,4.5))
tInd, TInd = 12, 12
print('PALM time: ' + str(tSeq[tInd]) + ', ' + 'WRF time: ' + str(time[TInd]))
plt.plot(uvSeq[tInd], zSeq, label='PALM', linewidth=1.0, color='r')
plt.plot(UV[TInd], Z[TInd], label='WRF', linewidth=1.0, color='b')
plt.xlabel(r"$\mathrm{\overline{u}_h}$ (m/s)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 0.0
xaxis_max = 16.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1200.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.6,0.86), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('t = ' + str(int(tSeq[tInd])) + 's')
fig.tight_layout() # adjust the layout
saveName = 'velo_pr_' + str(time[TInd]//3600) + 'hr' + '.png'
saveDir = '/scratch/projects/EERA-SP3/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" profile of theta """
fig, ax = plt.subplots(figsize=(4.5,4.5))
tInd, TInd = 3, 3
print('PALM time: ' + str(tSeq[tInd]) + ', ' + 'WRF time: ' + str(time[TInd]))
plt.plot(thetaSeq[tInd], zSeq, label='PALM', linewidth=1.0, color='r')
plt.plot(THETA[TInd]+300, Z[TInd], label='WRF', linewidth=1.0, color='b')
plt.xlabel(r"$\mathrm{\theta}$ (K)", fontsize=12)
plt.ylabel('z (m)', fontsize=12)
xaxis_min = 285.0
xaxis_max = 305.0
xaxis_d = 5.0
yaxis_min = 0.0
yaxis_max = 1200.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.6,0.86), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('t = ' + str(int(tSeq[tInd])) + 's')
fig.tight_layout() # adjust the layout
saveName = 'theta_pr_' + str(time[TInd]//3600) + 'hr' + '.png'
saveDir = '/scratch/projects/EERA-SP3/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
#plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()