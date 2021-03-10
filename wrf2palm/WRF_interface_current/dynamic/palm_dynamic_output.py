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
'''
This file creates and writes the dynamic driver netcdf file based on preprepared
transformed and interpolated wrf and camx files.
'''

import os
import time
import numpy as np
import netCDF4
from palm_wrf_utils import palm_wrf_gw


def palm_dynamic_output(wrf_files, interp_files, camx_interp_fname, dynamic_driver_file, times_sec,
                        dimensions, z_levels, z_levels_stag, ztop,
                        z_soil_levels, dx, dy, lon_center, lat_center,
                        rad_times_proc, rad_values_proc, sma, nested_domain):

    print('Processing interpolated files to dynamic driver')
    # dimension of the time coordinate
    dimtimes = len(times_sec)
    # other coordinates
    dimnames = ['z', 'zw', 'zsoil', 'x','xu', 'y', 'yv']                # z  height agl in m, zw staggered, zsoil 4 lev from wrf 
    dimsize_names = ['zdim' , 'zwdim', 'zsoildim', 'xdim', 'xudim', 'ydim', 'yvdim']    # palm: zw = z - 1
    x = np.arange(dx, dimensions['xdim']*dx+dx, dx)     # clean this
    print('dimension of x:', len(x))
    y = np.arange(dy, dimensions['ydim']*dy+dy, dy)     # clean this
    print('dimension of y:', len(y))
    # fill values
    fillvalue_float = float(-9999.0)
    # check out file and remove for sure
    try:
        os.remove(dynamic_driver_file)
    except:
        pass
    # create netcdf out file
    print('Driver file:', dynamic_driver_file)
    outfile = netCDF4.Dataset(dynamic_driver_file, "w", format="NETCDF4" )
    try:
        # time dimension and variable
        outfile.createDimension('time', dimtimes)
        _val_times =  outfile.createVariable('time',"f4", ("time"))
        # other dimensions and corresponding variables
        for _dim in zip(dimnames, dimsize_names):  #range (len(dimnames)):
            print(_dim[0],_dim[1], dimensions[_dim[1]])
            outfile.createDimension(_dim[0], dimensions[_dim[1]])
        _val_z_levels = outfile.createVariable('z',"f4", ("z"))
        _val_z_levels_stag = outfile.createVariable('zw',"f4", ("zw"))
        _val_z_soil_levels = outfile.createVariable('zsoil',"f4", ("zsoil"))
        
        # original code # !!!! bug2 we need 'x', 'y' and also 'ux', 'vy'
        # _val_y = outfile.createVariable('y',"f4", ("y"))
        # _val_x = outfile.createVariable('x',"f4", ("x"))
        
        _val_yv = outfile.createVariable('yv',"f4", ("yv"))
        _val_xu = outfile.createVariable('xu',"f4", ("xu"))
        _val_y = outfile.createVariable('y',"f4", ("y"))
        _val_x = outfile.createVariable('x',"f4", ("x"))

        # prepare influx/outflux area sizes
        zstag_all = np.r_[0., z_levels_stag, ztop]
        zwidths = zstag_all[1:] - zstag_all[:-1]
        print('zwidths', zwidths)
        areas_xb = np.zeros((len(z_levels), 1))
        areas_xb[:,0] = zwidths * dy
        areas_yb = np.zeros((len(z_levels), 1))
        areas_yb[:,0] = zwidths * dx
        areas_zb = dx*dy
        area_boundaries = (areas_xb.sum()*dimensions['ydim']*2
                + areas_yb.sum()*dimensions['xdim']*2
                + areas_zb*dimensions['xdim']*dimensions['ydim'])

        # write values for coordinates
        _val_times[:] = times_sec[:]
        _val_z_levels[:] = z_levels[:]
        _val_z_levels_stag[:] = z_levels_stag[:]
        _val_z_soil_levels[:] = z_soil_levels[:]
        
        # original code # !!!! bug2
        # _val_y[:] = y[:]
        # _val_x[:] = x[:]
        
        # print(dimensions['xdim'], dimensions['xudim'], dimensions['ydim'], dimensions['yvdim'])
        yv = np.arange(dy, dimensions['yvdim']*dy+dy, dy)
        xu = np.arange(dx, dimensions['xudim']*dx+dx, dx)
        y = np.arange(dy, dimensions['ydim']*dy+dy, dy) - dy/2
        x = np.arange(dx, dimensions['xdim']*dx+dx, dx) - dx/2
        _val_yv[:] = yv[:]
        _val_xu[:] = xu[:]
        _val_y[:] = y[:]
        _val_x[:] = x[:]

        # initialization of the variables and setting of init_* variables
        print("Processing initialization from file", interp_files[0])
        # open corresponding
        infile = netCDF4.Dataset(interp_files[0], "r", format="NETCDF4")
        try:
            # open variables in the input file
            init_atmosphere_pt = infile.variables['init_atmosphere_pt']
            init_atmosphere_qv = infile.variables['init_atmosphere_qv']
            init_atmosphere_u = infile.variables['init_atmosphere_u']
            init_atmosphere_v = infile.variables['init_atmosphere_v']
            init_atmosphere_w = infile.variables['init_atmosphere_w']
            init_soil_m = infile.variables['init_soil_m']
            init_soil_t = infile.variables['init_soil_t']

            # create netcdf structure
            _val_init_atmosphere_pt = outfile.createVariable('init_atmosphere_pt', "f4", ("z", "y", "x"),
                                                             fill_value=fillvalue_float)
            _val_init_atmosphere_pt.setncattr('lod', 2)
            _val_init_atmosphere_qv = outfile.createVariable('init_atmosphere_qv', "f4", ("z", "y", "x"),
                                                             fill_value=fillvalue_float)
            _val_init_atmosphere_qv.setncattr('lod', 2)
            _val_init_atmosphere_u = outfile.createVariable('init_atmosphere_u', "f4", ("z", "y", "xu"),
                                                            fill_value=fillvalue_float)
            _val_init_atmosphere_u.setncattr('lod', 2)
            _val_init_atmosphere_v = outfile.createVariable('init_atmosphere_v', "f4", ("z", "yv", "x"),
                                                            fill_value=fillvalue_float)
            _val_init_atmosphere_v.setncattr('lod', 2)
            _val_init_atmosphere_w = outfile.createVariable('init_atmosphere_w', "f4", ("zw", "y", "x"),
                                                            fill_value=fillvalue_float)
            _val_init_atmosphere_w.setncattr('lod', 2)
            _val_init_soil_t = outfile.createVariable('init_soil_t', "f4", ("zsoil", "y", "x"), fill_value=fillvalue_float)
            _val_init_soil_t.setncattr('lod', 2)
            _val_init_soil_m = outfile.createVariable('init_soil_m', "f4", ("zsoil", "y", "x"), fill_value=fillvalue_float)
            _val_init_soil_m.setncattr('lod', 2)
            # time dependent variables
            if not nested_domain:
                # SURFACE PRESSURE
                _val_surface_forcing_surface_pressure = outfile.createVariable('surface_forcing_surface_pressure', "f4",
                                                                               ("time"))
                # BOUNDARY - vertical slices from left, right, south, north, top
                varname = 'pt'
                _val_ls_forcing_pt_left = outfile.createVariable('ls_forcing_left_' + varname, "f4", ("time", "z", "y"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_pt_right = outfile.createVariable('ls_forcing_right_' + varname, "f4", ("time", "z", "y"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_pt_south = outfile.createVariable('ls_forcing_south_' + varname, "f4", ("time", "z", "x"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_pt_north = outfile.createVariable('ls_forcing_north_' + varname, "f4", ("time", "z", "x"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_pt_top = outfile.createVariable('ls_forcing_top_' + varname, "f4", ("time", "y", "x"),
                                                                fill_value=fillvalue_float)

                varname = 'qv'
                _val_ls_forcing_qv_left = outfile.createVariable('ls_forcing_left_' + varname, "f4", ("time", "z", "y"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_qv_right = outfile.createVariable('ls_forcing_right_' + varname, "f4", ("time", "z", "y"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_qv_south = outfile.createVariable('ls_forcing_south_' + varname, "f4", ("time", "z", "x"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_qv_north = outfile.createVariable('ls_forcing_north_' + varname, "f4", ("time", "z", "x"),
                                                                  fill_value=fillvalue_float)
                _val_ls_forcing_qv_top = outfile.createVariable('ls_forcing_top_' + varname, "f4", ("time", "y", "x"),
                                                                fill_value=fillvalue_float)

                varname = 'u'
                _val_ls_forcing_u_left = outfile.createVariable('ls_forcing_left_' + varname, "f4", ("time", "z", "y"),
                                                                fill_value=fillvalue_float)
                _val_ls_forcing_u_right = outfile.createVariable('ls_forcing_right_' + varname, "f4", ("time", "z", "y"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_u_south = outfile.createVariable('ls_forcing_south_' + varname, "f4", ("time", "z", "xu"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_u_north = outfile.createVariable('ls_forcing_north_' + varname, "f4", ("time", "z", "xu"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_u_top = outfile.createVariable('ls_forcing_top_' + varname, "f4", ("time", "y", "xu"),
                                                               fill_value=fillvalue_float)

                varname = 'v'
                _val_ls_forcing_v_left = outfile.createVariable('ls_forcing_left_' + varname, "f4", ("time", "z", "yv"),
                                                                fill_value=fillvalue_float)
                _val_ls_forcing_v_right = outfile.createVariable('ls_forcing_right_' + varname, "f4", ("time", "z", "yv"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_v_south = outfile.createVariable('ls_forcing_south_' + varname, "f4", ("time", "z", "x"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_v_north = outfile.createVariable('ls_forcing_north_' + varname, "f4", ("time", "z", "x"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_v_top = outfile.createVariable('ls_forcing_top_' + varname, "f4", ("time", "yv", "x"),
                                                               fill_value=fillvalue_float)

                varname = 'w'
                _val_ls_forcing_w_left = outfile.createVariable('ls_forcing_left_' + varname, "f4", ("time", "zw", "y"),
                                                                fill_value=fillvalue_float)
                _val_ls_forcing_w_right = outfile.createVariable('ls_forcing_right_' + varname, "f4", ("time", "zw", "y"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_w_south = outfile.createVariable('ls_forcing_south_' + varname, "f4", ("time", "zw", "x"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_w_north = outfile.createVariable('ls_forcing_north_' + varname, "f4", ("time", "zw", "x"),
                                                                 fill_value=fillvalue_float)
                _val_ls_forcing_w_top = outfile.createVariable('ls_forcing_top_' + varname, "f4", ("time", "y", "x"),
                                                               fill_value=fillvalue_float)

                # geostrophic wind
                _val_ls_forcing_ug = outfile.createVariable('ls_forcing_ug', "f4", ("time", "z"),
                                                            fill_value=fillvalue_float)
                _val_ls_forcing_vg = outfile.createVariable('ls_forcing_vg', "f4", ("time", "z"),
                                                            fill_value=fillvalue_float)
                time.sleep(1)

            # write values for initialization variables
            _val_init_atmosphere_pt[:, :, :] = init_atmosphere_pt[0, :, :, :]
            del init_atmosphere_pt # !!!! bug3, save memory
            _val_init_atmosphere_qv[:, :, :] = init_atmosphere_qv[0, :, :, :]
            del init_atmosphere_qv # !!!! bug3, save memory
            _val_init_atmosphere_u[:, :, :] = init_atmosphere_u[0, :, :, 1:]
            del init_atmosphere_u # !!!! bug3, save memory
            _val_init_atmosphere_v[:, :, :] = init_atmosphere_v[0, :, 1:, :]
            del init_atmosphere_v # !!!! bug3, save memory
            _val_init_atmosphere_w[:, :, :] = init_atmosphere_w[0, :, :, :]
            del init_atmosphere_w # !!!! bug3, save memory
            _val_init_soil_t[:, :, :] = init_soil_t[0, :, :, :]
            for k in range(0,_val_init_soil_m.shape[0]):
                # adjust soil moisture according soil_moisture_adjust field (if exists)
                _val_init_soil_m[k, :, :] = init_soil_m[0, k, :, :] * sma[:, :]
        finally:
            # close interpolated file
            infile.close()
        #
        if not nested_domain:
            # cycle over all included time steps
            for ts in range(0, len(interp_files)):
                print("Processing file",interp_files[ts])
                # open corresponding interpolated file
                infile = netCDF4.Dataset(interp_files[ts], "r", format="NETCDF4")
                try:
                    # open variables in the input file
                    init_atmosphere_pt = infile.variables['init_atmosphere_pt']
                    init_atmosphere_qv = infile.variables['init_atmosphere_qv']
                    init_atmosphere_u = infile.variables['init_atmosphere_u']
                    init_atmosphere_v = infile.variables['init_atmosphere_v']
                    init_atmosphere_w = infile.variables['init_atmosphere_w']
                    surface_forcing_surface_pressure = infile.variables['surface_forcing_surface_pressure']

                    ##################################
                    # write values for time dependent values
                    # surface pressure
                    _val_surface_forcing_surface_pressure[ts] = np.average(surface_forcing_surface_pressure[:,:,:], axis = (1,2))[0]
                    del surface_forcing_surface_pressure # !!!! bug3, save memory
                    # boundary conditions
                    _val_ls_forcing_pt_left[ts, :, :] = init_atmosphere_pt[0, :, :, 0]
                    _val_ls_forcing_pt_right[ts, :, :] = init_atmosphere_pt[0, :, :, dimensions['xdim'] - 1]
                    _val_ls_forcing_pt_south[ts, :, :] = init_atmosphere_pt[0, :, 0, :]
                    _val_ls_forcing_pt_north[ts, :, :] = init_atmosphere_pt[0, :, dimensions['ydim'] - 1, :]
                    _val_ls_forcing_pt_top[ts, :, :] = init_atmosphere_pt[0, dimensions['zdim'] - 1, :, :]
                    del init_atmosphere_pt # !!!! bug3, save memory
                    
                    _val_ls_forcing_qv_left[ts, :, :] = init_atmosphere_qv[0, :, :, 0]
                    _val_ls_forcing_qv_right[ts, :, :] = init_atmosphere_qv[0, :, :, dimensions['xdim'] - 1]
                    _val_ls_forcing_qv_south[ts, :, :] = init_atmosphere_qv[0, :, 0, :]
                    _val_ls_forcing_qv_north[ts, :, :] = init_atmosphere_qv[0, :, dimensions['ydim'] - 1, :]
                    _val_ls_forcing_qv_top[ts, :, :] = init_atmosphere_qv[0, dimensions['zdim'] - 1, :, :]
                    del init_atmosphere_qv # !!!! bug3, save memory

                    # Perform mass balancing
                    uxleft = init_atmosphere_u[0, :, :, 0]
                    uxright = init_atmosphere_u[0, :, :, dimensions['xdim'] - 1]
                    vysouth = init_atmosphere_v[0, :, 0, :]
                    vynorth = init_atmosphere_v[0, :, dimensions['ydim'] - 1, :]
                    wztop = init_atmosphere_w[0, dimensions['zwdim'] - 1, :, :]
                    mass_disbalance = ((uxleft * areas_xb).sum()
                        - (uxright * areas_xb).sum()
                        + (vysouth * areas_yb).sum()
                        - (vynorth * areas_yb).sum()
                        - (wztop * areas_zb).sum())
                    mass_corr_v = mass_disbalance / area_boundaries
                    print('Mass disbalance: {0:8g} m3/s (avg = {1:8g} m/s)'.format(
                        mass_disbalance, mass_corr_v))
                    uxleft -= mass_corr_v
                    uxright += mass_corr_v
                    vysouth -= mass_corr_v
                    vynorth += mass_corr_v
                    wztop += mass_corr_v

                    # Verify mass balance (optional)
                    #mass_disbalance = ((uxleft * areas_xb).sum()
                    #    - (uxright * areas_xb).sum()
                    #    + (vysouth * areas_yb).sum()
                    #    - (vynorth * areas_yb).sum()
                    #    - (wztop * areas_zb).sum())
                    #mass_corr_v = mass_disbalance / area_boundaries
                    #print('Mass balanced:   {0:8g} m3/s (avg = {1:8g} m/s)'.format(
                    #    mass_disbalance, mass_corr_v))

                    _val_ls_forcing_u_left[ts, :, :] = uxleft
                    _val_ls_forcing_u_right[ts, :, :] = uxright
                    _val_ls_forcing_u_south[ts, :, :] = init_atmosphere_u[0, :, 0, 1:]
                    _val_ls_forcing_u_north[ts, :, :] = init_atmosphere_u[0, :, dimensions['ydim'] - 1, 1:]
                    _val_ls_forcing_u_top[ts, :, :] = init_atmosphere_u[0, dimensions['zdim'] - 1, :, 1:]
                    del init_atmosphere_u # !!!! bug3, save memory

                    _val_ls_forcing_v_left[ts, :, :] = init_atmosphere_v[0, :, 1:, 0]
                    _val_ls_forcing_v_right[ts, :, :] = init_atmosphere_v[0, :, 1:, dimensions['xdim'] - 1]
                    _val_ls_forcing_v_south[ts, :, :] = vysouth
                    _val_ls_forcing_v_north[ts, :, :] = vynorth
                    _val_ls_forcing_v_top[ts, :, :] = init_atmosphere_v[0, dimensions['zdim'] - 1, 1:, :]
                    del init_atmosphere_v # !!!! bug3, save memory

                    _val_ls_forcing_w_left[ts, :, :] = init_atmosphere_w[0, :, :, 0]
                    _val_ls_forcing_w_right[ts, :, :] = init_atmosphere_w[0, :, :, dimensions['xdim'] - 1]
                    _val_ls_forcing_w_south[ts, :, :] = init_atmosphere_w[0, :, 0, :]
                    _val_ls_forcing_w_north[ts, :, :] = init_atmosphere_w[0, :, dimensions['ydim'] - 1, :]
                    _val_ls_forcing_w_top[ts, :, :] = wztop
                    del init_atmosphere_w # !!!! bug3, save memory

                finally:
                    # close interpolated file
                    infile.close()

                # write geostrophic wind
                print('Open wrf file '+wrf_files[ts])
                nc_wrf = netCDF4.Dataset(wrf_files[ts], 'r')
                try:
                    ug, vg = palm_wrf_gw(nc_wrf, lon_center, lat_center, z_levels)
                    _val_ls_forcing_ug[ts, :] = ug
                    _val_ls_forcing_vg[ts, :] = vg
                finally:
                    nc_wrf.close()

            # Write chemical boundary conds
            if camx_interp_fname:
                f_camx = netCDF4.Dataset(camx_interp_fname)
                try:
                    for vname, vval in f_camx.variables.items():
                        # PALM doesn't support 3D LOD=2 init for chem yet, we have to average the field
                        var = outfile.createVariable('init_atmosphere_'+vname,
                                'f4', ('z',), fill_value=fillvalue_float)
                        var.units = vval.units
                        var.lod = 1
                        var[:] = vval[0,:,:,:].mean(axis=(1,2))

                        var = outfile.createVariable('ls_forcing_left_'+vname,
                                'f4', ('time','z','y'), fill_value=fillvalue_float)
                        var.units = vval.units
                        var[:] = vval[:,:,:,0]

                        var = outfile.createVariable('ls_forcing_right_'+vname,
                                'f4', ('time','z','y'), fill_value=fillvalue_float)
                        var.units = vval.units
                        var[:] = vval[:,:,:,-1]

                        var = outfile.createVariable('ls_forcing_south_'+vname,
                                'f4', ('time','z','x'), fill_value=fillvalue_float)
                        var.units = vval.units
                        var[:] = vval[:,:,0,:]

                        var = outfile.createVariable('ls_forcing_north_'+vname,
                                'f4', ('time','z','x'), fill_value=fillvalue_float)
                        var.units = vval.units
                        var[:] = vval[:,:,-1,:]

                        var = outfile.createVariable('ls_forcing_top_'+vname,
                                'f4', ('time','y','x'), fill_value=fillvalue_float)
                        var.units = vval.units
                        var[:] = vval[:,-1,:,:]
                finally:
                    f_camx.close()

        if len(rad_times_proc) > 0:
            # process radiation inputs
            # radiation time dimension and variable
            outfile.createDimension('time_rad', len(rad_times_proc))
            _val_times =  outfile.createVariable('time_rad',"f4", ("time_rad"))
            _val_times[:] = rad_times_proc[:]
            # radiation variables
            var = outfile.createVariable('rad_sw_in', "f4", ("time_rad"), fill_value=fillvalue_float)
            var.setncattr('lod', 1)
            var.units = 'W/m2'
            var[:] = rad_values_proc[0][:]
            var = outfile.createVariable('rad_lw_in', "f4", ("time_rad"), fill_value=fillvalue_float)
            var.setncattr('lod', 1)
            var.units = 'W/m2'
            var[:] = rad_values_proc[1][:]
            var = outfile.createVariable('rad_sw_in_dif', "f4", ("time_rad"), fill_value=fillvalue_float)
            var.setncattr('lod', 1)
            var.units = 'W/m2'
            var[:] = rad_values_proc[2][:]

    finally:
        outfile.close()
    




