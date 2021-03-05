#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------#
#
# Scripts for processing of WRF and CAMx files to PALM dynamic driver
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2018-2021 Institute of Computer Science
#                     of the Czech Academy of Sciences, Prague
# Authors: Krystof Eben, Jaroslav Resler, Pavel Krc
#
#------------------------------------------------------------------------------#
'''Scripts for processing of WRF and CAMx files to PALM dynamic driver.

Usage: palm_dynamic -c <config_name> [-w]

Script requires name of the configuration on command line.
The corresponding case configuration files are placed in subdirectory
"configuration" and the are named <config_name>.conf. The template
of the config file is in the file palm_dynamic_defaults.py, the values
which agree with defaults need not be present in the user config.
The file palm_dynamic_init.py contains setting and calculation of
standard initialization values for particular system and can be adjusted.
The optional parameter -w allows to skip horizontal and vertical
interpolation in case it is already done.
'''
__version__ = '3.1'

import sys
import getopt
import os.path
from datetime import datetime, timedelta
import numpy as np
from pyproj import Proj, transform
import glob

import netCDF4

import palm_dynamic_config

##################################
# read configuration from command
# line and parse it
def print_help():
    print()
    print(__doc__)
    print('Version: '+__version__)

# set commandline variables
configname = 'test' # !!!!
wrf_interpolation = True
if sys.argv[0].endswith('pydevconsole.py'):
    # we are in Pycharm developlent console - set testing data
    configname = 'evropska_12s_basecase'
else:
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hc:w)",["help=","config=","write="])
        #print(opts)
    except getopt.GetoptError as err:
        print("Error:", err)
        print_help()
        sys.exit(2)
    # parse options
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif opt in ("-c", "--config"):
            configname = arg
        elif opt in ("-w", "--write"):
            wrf_interpolation = False

if configname == '':
    # script requires config name
    print("This script requires input parameter -c <config_name>")
    print_help()
    exit(2)

# Load config defaults, config file and standard init into the config module
palm_dynamic_config.configure(configname)
# Import values (including loaded) from config module into globals
from palm_dynamic_config import *

# Other modules are imported *AFTER* the config module has been filled with
# values (so that they can correctly import * from palm_dynamic_config).
import palm_wrf_utils
from palm_dynamic_output import palm_dynamic_output
import palm_dynamic_camx


print('Domain name: ', domain)
print('Resolution name: ', resolution)
print('Scenario name: ', scenario)
print('Read domain from static:', grid_from_static)
if grid_from_static:
    print('STATIC driver input file: ', static_driver_file)
print('WRF and CAMx file path:', wrf_dir_name)
print('WRF file mask:', wrf_file_mask)
print('CAMx file mask:', camx_file_mask)
print('Simulation start time:', origin_time)
print('Simulation hours:', simulation_hours)


###########################
# domain parameters

if grid_from_static:
    # get parameters of the horizontal domain from PALM STATIC driver
    try:
        ncs = netCDF4.Dataset(static_driver_file, "r", format="NETCDF4")
    except Exception as err:
        print("Error opening static driver file:")
        print(static_driver_file)
        print(err)
        sys.exit(1)
    # get horizontal structure of the domain
    nx = ncs.dimensions['x'].size
    ny = ncs.dimensions['y'].size
    dx = ncs.variables['x'][:][1] - ncs.variables['x'][:][0]
    dy = ncs.variables['y'][:][1] - ncs.variables['y'][:][0]
    origin_x = ncs.getncattr('origin_x')
    origin_y = ncs.getncattr('origin_y')
    origin_z = ncs.getncattr('origin_z')
    if origin_time == '':
        # origin_time is not prvided in configuration - read it from static driver
        origin_time = ncs.getncattr('origin_time')

    # create vertical structure of the domain
    # calculate dz in case dz is not supplied (dz<=0)
    if dz <= 0.0:
        print("dz = 0.0: set dz = dx")
        dz = dx
    if nz <= 0:
        print("nz > 0 needs to be set in config")
        sys.exit(1)

    # read terrain height (relative to origin_z) and
    # calculate and check the height of the surface canopy layer
    if 'zt' in ncs.variables.keys():
        terrain_rel = ncs.variables['zt'][:]
    else:
        terrain_rel = np.zeros([ny,nx])
    th = np.ceil(terrain_rel / dz)
    # building height
    if 'buildings_3d' in ncs.variables.keys():
        print(np.argmax(a != 0, axis=0))
        bh3 = ncs.variables['buildings_3d'][:]
        # minimum index of nonzeo value along inverted z
        bh = np.argmax(bh3[::-1], axis=0)
        # inversion back and masking grids with no buildings
        bh = bh3.shape[0] - bh
        bh[np.max(bh3, axis=0) == 0] = 0
    elif 'buildings_2d' in ncs.variables.keys():
        bh = ncs.variables['buildings_2d'][:]
        bh[bh.mask] = 0
        bh = np.ceil(bh / dz)
    else:
        bh = np.zeros([ny,nx])
    # plant canopy height
    if 'lad' in ncs.variables.keys():
        lad3 = ncs.variables['lad'][:]
        # replace non-zero values with 1
        lad3[lad3 != 0] = 1
        # minimum index of nonzeo value along inverted z
        lad = np.argmax(lad3[::-1], axis=0)
        # inversion back and masking grids with no buildings
        lad = lad3.shape[0] - lad
        lad[np.max(lad3, axis=0) == 0] = 0
    else:
        lad = np.zeros([ny,nx])
    # calculate maximum of surface canopy layer
    nscl = max(np.amax(th+bh),np.amax(th+lad))
    # check nz with ncl
    if nz < nscl + nscl_free:
        print('nz has to be higher than ', nscl + nscl_free)
        print('nz=', nz, ', dz=', dz, ', number of scl=', nscl, ',nscl_free=', nscl_free)
    if dz_stretch_level <= (nscl + nscl_free) * dz:
        print('stretching has to start in higher level than ', (nscl + nscl_free) * dz)
        print('dz_stretch_level=', dz_stretch_level, ', nscl=', nscl, ', nscl_free=', nscl_free, ', dz=', dz)
    if 'soil_moisture_adjust' in ncs.variables.keys():
        soil_moisture_adjust = ncs.variables['soil_moisture_adjust'][:]
    else:
        soil_moisture_adjust = np.ones(shape=(ny, nx), dtype=float)
    # close static driver nc file
    ncs.close()
else:
    #TODO check all necessary parameters are set and are correct
    #
    # initialize necessary arrays
    try:
        terrain = np.zeros(shape=(ny, nx), dtype=float)
        soil_moisture_adjust = np.ones(shape=(ny, nx), dtype=float)
    except:
        print('domain parameters nx, ny have to be set in configuration')
        sys.exit(1)

# absolute terrain needed for vertical interpolation of wrf data
terrain = terrain_rel + origin_z

# print domain parameters and check ist existence in caso of setup from config
try:
    print('Domain parameters:')
    print('nx, ny, nz:', nx, ny, nz)
    print('dx, dy, dz:', dx, dy, dz)
    print('origin_x, origin_y:', origin_x, origin_y)
    print('Base of domain is in level origin_z:', origin_z)
except:
    print('domain parameters have to be read from static driver or set in configuration')
    sys.exit(1)

# centre of the domain (needed for ug,vg calculation)
xcent = origin_x + nx * dx / 2.0
ycent = origin_y + ny * dy / 2.0
# WGS84 projection for transformation to lat-lon
inproj = Proj('+init='+proj_palm)
print('inproj', inproj)
lonlatproj = Proj('+init='+proj_wgs84)
print('lonlatproj', lonlatproj)
cent_lon, cent_lat = transform(inproj, lonlatproj, xcent, ycent)
print('xcent, ycent:',xcent, ycent)
print('cent_lon, cent_lat:', cent_lon, cent_lat)
# prepare target grid
irange = origin_x + dx * (np.arange(nx, dtype='f8') + .5)
jrange = origin_y + dy * (np.arange(ny, dtype='f8') + .5)
palm_grid_y, palm_grid_x = np.meshgrid(jrange, irange, indexing='ij')
palm_grid_lon, palm_grid_lat = transform(inproj, lonlatproj, palm_grid_x, palm_grid_y)

######################################
# build structure of vertical layers
# remark:
# PALM input requires nz=ztop in PALM
# but the output file in PALM has max z higher than z in PARIN.
# The highest levels in PALM are wrongly initialized !!!
#####################################
# fill out z_levels
z_levels = np.zeros(nz,dtype=float)
z_levels_stag = np.zeros(nz-1,dtype=float)
dzs = dz
z_levels[0] = dzs/2.0
for i in range(nz-1):
    z_levels[i+1] = z_levels[i] + dzs
    z_levels_stag[i] = (z_levels[i+1]+z_levels[i])/2.0
    dzso = dzs
    if z_levels[i+1] + dzs >= dz_stretch_level:
        dzs = min(dzs * dz_stretch_factor, dz_max)
ztop = z_levels[-1] + dzs / 2.
print('z:',z_levels)
print('zw:',z_levels_stag)

######################################
# get time extent of the PALM simulation
#####################################
# get complete list of wrf files
wrf_file_list = glob.glob(os.path.join(wrf_dir_name, wrf_file_mask))
# get simulation origin and final time as datetime
start_time = datetime.strptime(origin_time, '%Y-%m-%d %H:%M:%S')
end_time = start_time + timedelta(hours=simulation_hours)
end_time_rad = end_time
print('PALM simulation extent', start_time, end_time, simulation_hours)
if nested_domain:
    print('Nested domain - process only initialization.')
    print('Set end_time = start_time')
    end_time = start_time

# get wrf times and sort wrf files by time
print('Analyse WRF files dates:')
file_times = []
for wrf_file in wrf_file_list:
    # get real time from wrf file
    nc_wrf = netCDF4.Dataset(wrf_file, "r", format="NETCDF4")
    ta = nc_wrf.variables['Times'][:]
    t = ta.tobytes().decode("utf-8")
    td = datetime.strptime(t, '%Y-%m-%d_%H:%M:%S')
    print(os.path.basename(wrf_file), ': ', td)
    file_times.append((td,wrf_file))
file_times.sort() # a list that contains all the times of each wrfout file
times = []
wrf_files = []
for tf in file_times:
    if end_time is None or tf[0] <= end_time:
        times.append(tf[0])
        wrf_files.append(tf[1])
#print('PALM output times:')
#print('\n'.join('{}'.format(t) for t in times))
print('PALM output times:', ', '.join('{}'.format(t) for t in times))

if not times.__contains__(start_time):
    print('WRF files does not contain PALM origin_time timestep - cannot process!')
    exit(1)

if not times.__contains__(end_time):
    print('WRF files does not contain PALM end_time timestep - cannot process!')
    exit(1)

start_index = times.index(start_time)
end_index = times.index(end_time)

if not nested_domain and end_index-start_index < simulation_hours:
    print('Number of WRF files does not aggre with number of simulation hours')
    exit(1)

print('PALM simulation timestep number are from ', start_index, ' to ', end_index, ' WRF file.')

# create list of processed wrf files
wrf_files_proc = []
for wf in wrf_files[start_index:end_index+1]:
    wrf_files_proc.append(wf)
# id of process files
simul_id = domain + "_d" + resolution

######################################
#VERTICAL AND HORIZONTAL INTERPOLATION
######################################
# get hydro and soil variables contained in wrf files
shvars = []
z_soil_levels = []
nc_wrf = netCDF4.Dataset(wrf_files_proc[0], "r", format="NETCDF4")
try:
    # hydro variables
    shvars = sorted(set(['QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP']).intersection(nc_wrf.variables.keys()))
    # soil layers
    if 'ZS' in nc_wrf.variables.keys():
        z_soil_levels = nc_wrf.variables['ZS'][0,:].data.tolist()
finally:
    nc_wrf.close()
print('Hydro variables in wrf files:', '+'.join(shvars))

print('Start of vertical and horizontal interpolation of inputs to the PALM domain.')
interp_files = []
regridder = None
for wrf_file in wrf_files_proc:
    print ("Input wrf file: ", wrf_file)
    wrf_file = wrf_files_proc[0] #!!!!
    hinterp_file = wrf_file+"_"+simul_id+'.hinterp'
    hinterp_log = wrf_file + "_" + simul_id + '.hinterp.log'
    vinterp_file = wrf_file+"_"+simul_id+'.interp'
    interp_files.append(vinterp_file)
    if wrf_interpolation:
        try:
             os.remove(vinterp_file)
             os.remove(hinterp_file)
             os.remove(hinterp_log)
        except:
              pass

        print('Horizontal interpolation...')
        f_wrf = netCDF4.Dataset(wrf_file, 'r')
        try:
            if not regridder:
                trans_wrf = palm_wrf_utils.WRFCoordTransform(f_wrf)
                palm_in_wrf_y, palm_in_wrf_x = trans_wrf.latlon_to_ji(
                                                palm_grid_lat, palm_grid_lon)
                regridder = palm_wrf_utils.BilinearRegridder(
                        palm_in_wrf_x, palm_in_wrf_y, preloaded=True)
                regridder_u = palm_wrf_utils.BilinearRegridder(
                        palm_in_wrf_x+.5, palm_in_wrf_y, preloaded=True)
                regridder_v = palm_wrf_utils.BilinearRegridder(
                        palm_in_wrf_x, palm_in_wrf_y+.5, preloaded=True)

            f_out = netCDF4.Dataset(hinterp_file, 'w', format='NETCDF4')
            try:
                # dimensions
                f_out.createDimension('Time', None)
                for d in 'bottom_top bottom_top_stag soil_layers_stag'.split():
                    f_out.createDimension(d, len(f_wrf.dimensions[d]))
                f_out.createDimension('west_east', len(irange))
                f_out.createDimension('south_north', len(jrange))

                # copied vars
                for varname in 'PH PHB HGT T W TSLB SMOIS MU MUB P PB PSFC'.split():
                    v_wrf = f_wrf.variables[varname]
                    v_out = f_out.createVariable(varname, 'f4', v_wrf.dimensions)
                    v_out[:] = regridder.regrid(v_wrf[...,regridder.ys,regridder.xs])

                # U and V have special treatment (unstaggering)
                v_out = f_out.createVariable('U', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
                v_out[:] = regridder_u.regrid(f_wrf.variables['U'][...,regridder_u.ys,regridder_u.xs])
                v_out = f_out.createVariable('V', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
                v_out[:] = regridder_v.regrid(f_wrf.variables['V'][...,regridder_v.ys,regridder_v.xs])

                # calculated SPECHUM
                v_out = f_out.createVariable('SPECHUM', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))
                vdata = regridder.regrid(f_wrf.variables[shvars[0]][...,regridder.ys,regridder.xs])
                for vname in shvars[1:]:
                    vdata += regridder.regrid(f_wrf.variables[vname][...,regridder.ys,regridder.xs])
                v_out[:] = vdata
            finally:
                f_out.close()
        finally:
            f_wrf.close()

        print('Vertical interpolation...')
        palm_wrf_utils.palm_wrf_vertical_interp(hinterp_file, vinterp_file, wrf_file, z_levels,
                                                z_levels_stag, z_soil_levels, origin_z, terrain,
                                                wrf_hybrid_levs, vinterp_terrain_smoothing)

if radiation_from_wrf:
    print('Start processing of radiation inputs from the WRF radiation files.')
    # get available times from wrf radiation output files and sort files by time
    # get complete list of wrf radiation files
    rad_file_list = glob.glob(os.path.join(wrf_dir_name, wrf_rad_file_mask))
    rad_file_times = []
    print('Analyse WRF radiation files dates:')
    for rad_file in rad_file_list:
        # get real time from wrf radiation file
        nc_rad = netCDF4.Dataset(rad_file, "r", format="NETCDF4")
        ta = nc_rad.variables['Times'][:]
        t = ta.tobytes().decode("utf-8")
        td = datetime.strptime(t, '%Y-%m-%d_%H:%M:%S')
        print(os.path.basename(rad_file), ': ', td)
        rad_file_times.append((td, rad_file))
    rad_file_times.sort()
    rad_times = []
    rad_files = []
    for tf in rad_file_times:
        if end_time_rad is None or tf[0] <= end_time_rad:
            rad_times.append(tf[0])
            rad_files.append(tf[1])
    # check list of available times
    if not rad_times.__contains__(start_time):
        print('WRF radiation files does not contain PALM origin_time timestep - cannot process!')
        exit(1)
    if not rad_times.__contains__(end_time_rad):
        print('WRF radiation files does not contain PALM end_time_rad timestep - cannot process!')
        exit(1)
    rad_start_index = rad_times.index(start_time)
    rad_end_index = rad_times.index(end_time_rad)
    print('PALM radiation timestep numbers are from ', rad_start_index, ' to ', rad_end_index, ' WRF radiation file.')
    # get radiation timestep
    rad_timestep = (rad_times[rad_start_index+1] - rad_times[rad_start_index]).total_seconds()
    print('Timestep of the wrf radiation data is ', rad_timestep,' seconds.')
    # check consistency of the series
    for i in range(rad_start_index, rad_end_index):
        if (rad_times[i+1] - rad_times[i]).total_seconds() != rad_timestep:
            print('Inconsistent timeline of radiation inputs! Cannot process.')
            print(rad_times[rad_start_index:rad_end_index])
            exit(1)
    # create list of processed wrf radiation files and times
    rad_files_proc = []
    rad_times_proc = []
    for i in range(rad_start_index, rad_end_index + 1):
        rad_files_proc.append(rad_files[i])
        rad_times_proc.append((rad_times[i]-rad_times[rad_start_index]).total_seconds())
    print("Radiation times are ", rad_times_proc)
    # process radiation inputs
    rad_swdown = []
    rad_lwdown = []
    rad_swdiff = []
    nfiles = 0
    for rad_file in rad_files_proc:
        print ("Input wrf radiation file: ", rad_file)
        ncf = netCDF4.Dataset(rad_file, "r", format="NETCDF4")
        try:
            if nfiles == 0:
                # create list of (i,j) indices used for smoothing of the radiation
                print('Build list of indices for radiation smoothig.')
                palmproj = Proj(init=proj_palm)
                lonlatproj = Proj(init=proj_wgs84)
                rad_ind = []
                for i in range(ncf.dimensions['west_east'].size):
                    for j in range(ncf.dimensions['south_north'].size):
                        # transfrom lat-lon to PALM domain in meters
                        lonij = ncf.variables['XLONG'][0,j,i]
                        latij = ncf.variables['XLAT'][0,j,i]
                        xij, yij = transform(lonlatproj, palmproj, lonij, latij)
                        if abs(xij-xcent)<=radiation_smoothing_distance and \
                            abs(yij-ycent)<=radiation_smoothing_distance:
                            rad_ind.append([i,j])
                ngrids = len(rad_ind)
            # process wrf radiation file
            nfiles += 1
            swdown = 0.0
            lwd = 0.0
            swddif = 0.0
            for ij in rad_ind:
                swdown = swdown + ncf.variables['SWDOWN'][0,ij[1],ij[0]]
                lwd    = lwd    + ncf.variables['GLW'][0,ij[1],ij[0]]
                swddif = swddif + ncf.variables['SWDDIF'][0,ij[1],ij[0]]
            rad_swdown.append(swdown / ngrids)
            rad_lwdown.append(lwd / ngrids)
            rad_swdiff.append(swddif / ngrids)
        finally:
            ncf.close()
    # create list of all radiation values
    rad_values_proc = [rad_swdown, rad_lwdown, rad_swdiff]
else:
    rad_times_proc = []
    rad_values_proc = []

camx_interp_fname = None
if not nested_domain:
    # process camx files
    camx_file_list = glob.glob(os.path.join(wrf_dir_name, camx_file_mask))
    if camx_file_list:
        print('Processing CAMx input files: {0}'.format(', '.join(camx_file_list)))
        camx_interp_fname = os.path.join(wrf_dir_name, 'CAMx_{0}_interp.nc'.format(simul_id))

        palm_dynamic_camx.process_files(camx_file_list, camx_interp_fname,
                palm_grid_lat, palm_grid_lon, terrain_rel, z_levels,
                times[start_index:end_index+1], species_names,
                camx_conversions, camx_helpers)

# ===================== CREATE NETCDF DRIVER ==============================
# calculate relative times from simulation start
times_sec = []
for t in times[start_index:end_index+1]:
    times_sec.append((t-start_time).total_seconds())
# collect dimension sizes
dimensions = {'zdim': nz, 'zwdim': nz-1, 'zsoildim': len(z_soil_levels), 'xdim': nx, 'xudim': nx-1, 'ydim': ny, 'yvdim': ny-1}
# process interpolated files to dynamic driver
palm_dynamic_output(wrf_files, interp_files, camx_interp_fname, dynamic_driver_file, times_sec, dimensions,
                    z_levels, z_levels_stag, ztop, z_soil_levels, dx, dy, cent_lon, cent_lat,
                    rad_times_proc, rad_values_proc, soil_moisture_adjust, nested_domain)

print('Creation of dynamic driver finished.')
