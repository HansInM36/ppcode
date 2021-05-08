import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

readDir = '/scratch/palmdata/JOBS/EERASP3_1/INPUT'
readName = "EERASP3_1_dynamic"

nx, ny, nz = 384, 384, 80
dx, dy, dz = 40, 40, 10

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)


tSeq = data.variables['time'][:]

zSeq = data.variables['z'][:]
zwSeq = data.variables['zw'][:]


pSeq = data.variables['surface_forcing_surface_pressure'][:]

init_atmosphere_pt = np.array(data.variables['init_atmosphere_pt'])
init_atmosphere_qv = np.array(data.variables['init_atmosphere_qv'])

init_atmosphere_u = np.array(data.variables['init_atmosphere_u'])
init_atmosphere_v = np.array(data.variables['init_atmosphere_v'])
init_atmosphere_w = np.array(data.variables['init_atmosphere_w'])

ls_forcing_ug = np.array(data.variables['ls_forcing_ug'])
ls_forcing_vg = np.array(data.variables['ls_forcing_vg'])

ls_forcing_left_pt = np.array(data.variables['ls_forcing_left_pt'])
ls_forcing_right_pt = np.array(data.variables['ls_forcing_right_pt'])
ls_forcing_south_pt = np.array(data.variables['ls_forcing_south_pt'])
ls_forcing_north_pt = np.array(data.variables['ls_forcing_north_pt'])
ls_forcing_top_pt = np.array(data.variables['ls_forcing_top_pt'])

ls_forcing_left_qv = np.array(data.variables['ls_forcing_left_qv'])
ls_forcing_right_qv = np.array(data.variables['ls_forcing_right_qv'])
ls_forcing_south_qv = np.array(data.variables['ls_forcing_south_qv'])
ls_forcing_north_qv = np.array(data.variables['ls_forcing_north_qv'])
ls_forcing_top_qv = np.array(data.variables['ls_forcing_top_qv'])

ls_forcing_left_u = np.array(data.variables['ls_forcing_left_u'])
ls_forcing_right_u = np.array(data.variables['ls_forcing_right_u'])
ls_forcing_south_u = np.array(data.variables['ls_forcing_south_u'])
ls_forcing_north_u = np.array(data.variables['ls_forcing_north_u'])
ls_forcing_top_u = np.array(data.variables['ls_forcing_top_u'])

ls_forcing_left_v = np.array(data.variables['ls_forcing_left_v'])
ls_forcing_right_v = np.array(data.variables['ls_forcing_right_v'])
ls_forcing_south_v = np.array(data.variables['ls_forcing_south_v'])
ls_forcing_north_v = np.array(data.variables['ls_forcing_north_v'])
ls_forcing_top_v = np.array(data.variables['ls_forcing_top_v'])

ls_forcing_left_w = np.array(data.variables['ls_forcing_left_w'])
ls_forcing_right_w = np.array(data.variables['ls_forcing_right_w'])
ls_forcing_south_w = np.array(data.variables['ls_forcing_south_w'])
ls_forcing_north_w = np.array(data.variables['ls_forcing_north_w'])
ls_forcing_top_w = np.array(data.variables['ls_forcing_top_w'])

data.close()

### write large-scale forcing data (constant)
writeDir = '/scratch/palmdata/JOBS/EERASP3_1_ndg_nogw/INPUT'
writeName = 'EERASP3_1_ndg_nogw_lsf'

tInd = 6
timeSkip = 21600.0

W = 0.0

itemList1 = ['time', 'shf', 'qsws', 'pt_surface', 'q_surface' ,'surface_pressure']
unitList1 = ['(s)', '(K m/s)', '(m/s kg/kg)', '(K)', '(kg/kg)', '(hPa)']

itemList2 = ['zu (m)', 'ug (m/s)', 'vg (m/s)', 'w_subs (m/s)', 'td_lsa_lpt (K/s)', 'td_lsa_q (kg/kgs)', 'td_sub_lpt (K/s)', 'td_sub_q (kg/kgs)']

out = open(writeDir + '/' + writeName,"w")
out.write("# dara obtained from " + readDir + '/' + readName + "\n")
out.write("#" + ('{:>18}'*len(itemList1)).format(*itemList1) + "\n")
out.write("#" + ('{:>18}'*len(unitList1)).format(*unitList1) + "\n")
for i in [0,1]:
    if i == 0:
        time = 0
    else:
        time = 999999999 
    tmp = []
    tmp += [str(np.round(time,2))]
    tmp += [str(0.0)] # no data for surface heat flux
    tmp += [str(0.0)] # no data for surface water flux
    tmp += [str(np.round(init_atmosphere_pt[0].mean(),4))] #[str(np.round(init_atmosphere_pt[0].mean(),4))] # no data for surface potential temperature, use the mean of initial
    tmp += [str(np.round(init_atmosphere_qv[0].mean(),6))] #[str(np.round(init_atmosphere_qv[0].mean(),6))] # no data for surface humidity, use the mean of initial      
    tmp += [str(np.round(pSeq[tInd]/100,4))]
    out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")

out.write("#" + ('{:>18}'*len(itemList2)).format(*itemList2) + "\n")
for i in [0,1]:
    if i == 0:
        time = 0
    else:
        time = 999999999    
    out.write("# " + str(np.round(time,2)) + "\n")            
    for zInd in range(zSeq.size+1):
        if zInd == zSeq.size: # add a top layer identical to the zSeq[-1] so that the PALM case can use the interpolated data at the domain top  
            tmp = []
            tmp += [str(np.round(zSeq[zInd-1]+10,4))]
            tmp += [str(np.round(0,6))] #[str(np.round(ls_forcing_ug[tInd,zInd],6))]
            tmp += [str(np.round(0,6))] #[str(np.round(ls_forcing_vg[tInd,zInd],6))]
            tmp += [str(np.round(0.0,6))] # no data for the large-scale vertical subsidence profile w_subs
            tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of temperature td_lsa_lpt
            tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of humudity td_lsa_q
            tmp += [str(np.round(0.0,7))] # no data for the large-scale subsidence tendencies of temperature td_sub_lpt
            tmp += [str(np.round(0.0,10))] # no data for the large-scale subsidence tendencies of humidity td_sub_q           
        else:
            tmp = []
            tmp += [str(np.round(zSeq[zInd],4))]
            tmp += [str(np.round(0,6))] #[str(np.round(ls_forcing_ug[tInd,zInd],6))]
            tmp += [str(np.round(0,6))] #[str(np.round(ls_forcing_vg[tInd,zInd],6))]
            tmp += [str(np.round(0.0,6))] # no data for the large-scale vertical subsidence profile w_subs
            tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of temperature td_lsa_lpt
            tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of humudity td_lsa_q
            tmp += [str(np.round(0.0,7))] # no data for the large-scale subsidence tendencies of temperature td_sub_lpt
            tmp += [str(np.round(0.0,10))] # no data for the large-scale subsidence tendencies of humidity td_sub_q
        out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")
    out.write("\n")
out.close()


### write large-scale forcing data (time-dependent)

writeDir = '/scratch/palmdata/JOBS/EERASP3_ndg/INPUT'
writeName = 'EERASP3_ndg_lsf'

tIndList = [6]
tLableList = [0.0]

timeSkip = 21600.0

W = 0.0

itemList1 = ['time', 'shf', 'qsws', 'pt_surface', 'q_surface' ,'surface_pressure']
unitList1 = ['(s)', '(K m/s)', '(m/s kg/kg)', '(K)', '(kg/kg)', '(hPa)']

itemList2 = ['zu (m)', 'ug (m/s)', 'vg (m/s)', 'w_subs (m/s)', 'td_lsa_lpt (K/s)', 'td_lsa_q (kg/kgs)', 'td_sub_lpt (K/s)', 'td_sub_q (kg/kgs)']

out = open(writeDir + '/' + writeName,"w")
out.write("# dara obtained from " + readDir + '/' + readName + "\n")
out.write("#" + ('{:>18}'*len(itemList1)).format(*itemList1) + "\n")
out.write("#" + ('{:>18}'*len(unitList1)).format(*unitList1) + "\n")
for tInd in tIndList:
    tmp = []
    tmp += [str(np.round(tSeq[tInd]-timeSkip,2))]
    tmp += [str(0.0)] # no data for surface heat flux
    tmp += [str(0.0)] # no data for surface water flux
    tmp += [str(np.round(init_atmosphere_pt[0].mean(),4))] # no data for surface potential temperature, use the mean of initial
    tmp += [str(np.round(init_atmosphere_qv[0].mean(),6))] # no data for surface humidity, use the mean of initial      
    tmp += [str(np.round(pSeq[tInd]/100,4))]
    out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")

out.write("#" + ('{:>18}'*len(itemList2)).format(*itemList2) + "\n")
for tInd in tIndList:
    out.write("# " + str(np.round(tSeq[tInd]-timeSkip,2)) + "\n")            
    for zInd in range(zSeq.size):
        tmp = []
        tmp += [str(np.round(zSeq[zInd],4))]
        tmp += [str(np.round(ls_forcing_ug[tInd,zInd],6))]
        tmp += [str(np.round(ls_forcing_vg[tInd,zInd],6))]
        tmp += [str(np.round(0.0,6))] # no data for the large-scale vertical subsidence profile w_subs
        tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of temperature td_lsa_lpt
        tmp += [str(np.round(0.0,10))] # no data for the horizontal large-scale advection tendencies of humudity td_lsa_q
        tmp += [str(np.round(0.0,7))] # no data for the large-scale subsidence tendencies of temperature td_sub_lpt
        tmp += [str(np.round(0.0,10))] # no data for the large-scale subsidence tendencies of humidity td_sub_q
        out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")
    out.write("\n")
out.close()