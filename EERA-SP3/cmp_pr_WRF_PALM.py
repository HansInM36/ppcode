import os
import sys
sys.path.append('/scratch/ppcode')
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.signal
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
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "wrfout_d01_2015-07-01_00:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

attr = lambda a: getattr(data, a)
#attr('TRUELAT1')

dimlist = list(data.dimensions)
varlist = list(data.variables)

time = data.variables['Times'][:6]
time.tobytes().decode("utf-8")

XLONG = data.variables['XLONG'][0]
XLAT = data.variables['XLAT'][0]
x, y = funcs.lonlat2cts(XLONG, XLAT) # this is for scalar
# interpolate to u position for only interval points (without the leftmost and rightmost points)
xu = (x[:,:-1] + x[:,1:]) / 2
yu = (y[:,:-1] + y[:,1:]) / 2
# interpolate to v position for only interval points (without the southmost and northmost points)
xv = (x[:-1,:] + x[1:,:]) / 2
yv = (y[:-1,:] + y[1:,:]) / 2

z = list(np.arange(0,25,10)) + [29,33] + list(np.arange(35,55,8)) + list(np.arange(60,250,15)) + \
    list(np.arange(300,400,50)) + list(np.arange(450,1000,100)) + list(np.arange(1300,3500,500)) + \
    list(np.arange(4000,18000,1500))
z = np.array(z)

U = data.variables['U'][126:139,:,:,1:-1]
V = data.variables['V'][126:139,:,1:-1,:]
T = data.variables['T'][126:139,:,:,:] + 300

HFX = data.variables['HFX'][:13,:,:]

T2 = data.variables['T2'][:13,:,:]
TH2 = data.variables['TH2'][:13,:,:]


""" FINO1 data """
### FINO1 site coordinates
lon = funcs.hms2std(6,35,15.5)
lat = funcs.hms2std(54,0,53.5)
x0, y0 = funcs.lonlat2cts(lon, lat)

### find the closest point to FINO1
Ju,Iu = 0, 0
Jv,Iv = 0, 0
J,I = 0, 0

# u position
Min = 1e10
for j in range(U.shape[2]):
    for i in range(U.shape[3]):
        d2 = (xu[j,i] - x0)**2 + (yu[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            Ju, Iu = j, i # J, I is the index of the point closest to the FINO1

# v position
Min = 1e10
for j in range(V.shape[2]):
    for i in range(V.shape[3]):
        d2 = (xv[j,i] - x0)**2 + (yv[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            Jv, Iv = j, i # J, I is the index of the point closest to the FINO1

# scalar position
Min = 1e10
for j in range(T.shape[2]):
    for i in range(T.shape[3]):
        d2 = (x[j,i] - x0)**2 + (y[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            J, I = j, i # J, I is the index of the point closest to the FINO1


U_wrf = U[:,:,Ju,Iu]
V_wrf = V[:,:,Jv,Iv]
T_wrf = T[:,:,J,I]
UV_wrf = np.sqrt(np.power(U_wrf,2) + np.power(V_wrf,2))
WD_wrf = funcs.wd(U_wrf,V_wrf)

""" WRFpp data """
readDir = '/scratch/projects/EERA-SP3/data/WRFpp'
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

U_wrfpp = U[126:139,:]
V_wrfpp = V[126:139,:]
T_wrfpp = THETA[126:139,:]
UV_wrfpp = np.sqrt(np.power(U_wrfpp,2) + np.power(V_wrfpp,2))
WD_wrfpp = funcs.wd(U_wrfpp,V_wrfpp)



""" PALM data """
jobName = 'EERASP3_2'
dir = '/scratch/palmdata/JOBS/' + jobName
u = pr_palm(dir, jobName, ['.000','.001','.002'], 'u')
v = pr_palm(dir, jobName, ['.000','.001','.002'], 'v')
theta = pr_palm(dir, jobName, ['.000','.001','.002'], 'theta')

uv = np.sqrt(np.power(u[2],2) + np.power(v[2],2))
wd = funcs.wd(u[2],v[2])

""" profile of horizontal velocity """
for tInd in range(13):
    fig, ax = plt.subplots(figsize=(3,4.5))
    print('PALM time: ' + str(u[0][tInd]) + ', ' + 'WRF time: ' + str(time[tInd]))
    plt.plot(uv[tInd], u[1], label='PALM', linewidth=1.0, color='b')
    plt.plot(UV_wrfpp[tInd], Z[tInd], label='WRFpp', marker='', linestyle='-', linewidth=1.0, color='r')
    plt.plot(UV_wrf[tInd], z, label='WRF', marker='', linestyle='-', linewidth=1.0, color='k')
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
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
    saveName = 'cmp_velo_pr_' + str(u[0][tInd]//3600) + 'hr' + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_velo_pr_2'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of horizontal wind direction """
for tInd in range(13):
    fig, ax = plt.subplots(figsize=(3,4.5))
    print('PALM time: ' + str(u[0][tInd]) + ', ' + 'WRF time: ' + str(time[tInd]))
    plt.plot(wd[tInd][1:], u[1][1:], label='PALM', linewidth=1.0, color='b')
    plt.plot(WD_wrfpp[tInd][1:], Z[tInd][1:], label='WRFpp', marker='', linestyle='-', linewidth=1.0, color='r')
    plt.plot(WD_wrf[tInd][1:], z[1:], label='WRF', marker='', linestyle='-', linewidth=1.0, color='k')
    plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 0.0
    xaxis_max = 360.0
    xaxis_d = 60
    yaxis_min = 0
    yaxis_max = 1000.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'cmp_wd_pr_' + str(u[0][tInd]//3600) + 'hr' + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_wd_pr'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of theta """
for tInd in range(13):
    fig, ax = plt.subplots(figsize=(3.0,4.5))
    print('PALM time: ' + str(u[0][tInd]) + ', ' + 'WRF time: ' + str(time[tInd]))
    plt.plot(T_wrf[tInd], z, label='WRF', linewidth=1.0, linestyle='-', color='k')
    plt.plot(theta[2][tInd], theta[1], label='PALM', linewidth=1.0, linestyle='-', color='b')
    plt.plot(T_wrfpp[tInd], Z[tInd], label='WRFpp', linewidth=1.0, linestyle='-', color='r')
    plt.xlabel(r"$\mathrm{\overline{\theta}}$ (K)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 20.0
    xaxis_max = 320.0
    xaxis_d = 100.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(int(u[0][tInd])) + 's')
    fig.tight_layout() # adjust the layout
#    saveName = 'cmp_theta_pr_' + str(u[0][tInd]//3600) + 'hr' + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_theta_pr'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()