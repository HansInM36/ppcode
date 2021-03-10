#!/usr/bin/python3
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
This file provides initialization of the basic variables and structures base on
the default and user config values. It can be adjusted to the needs of particular
user system or structure of the data storage.
'''

import os
from pathlib import Path
import numpy as np

# paths of directories
dir_base = os.path.abspath(Path(dir_scripts).parent.parent)
print('Base dir:', dir_base)
dir_in = os.path.join(dir_base, domain)
print('Input dir:', dir_in)
if scenario == '':
    dir_in_scen = dir_in
else:
    dir_in_scen = os.path.join(dir_in,scenario)
    print('Scenario input dir:', dir_in_scen)
dir_out = os.path.join(dir_base, domain)
print('Output dir:', dir_out)

# file names of PALM PIDS drivers
# extension of file name for scenario
scenario_ext = ("_"+scenario if scenario != "" else "")
# static driver netcdf file name
if static_driver_file == "":
    static_driver_file = os.path.join(dir_out, domain+"_static_driver"+"_d"+resolution+scenario_ext+".nc")
# dynamic driver netcdf file name
if dynamic_driver_file == "":
    dynamic_driver_file = os.path.join(dir_out, domain+"_dynamic_driver"+"_d"+resolution+scenario_ext+".nc")

# parameters of dynamic driver
# minimal number of free surface canopy layers above top of terrain with building and plant canopy
nscl_free = 3
# default path of wrf files in case it is not set in user config
if wrf_dir_name == '':
    wrf_dir_name = os.path.join(dir_in, 'wrf')

# Settings for geostrophic wind
gw_gfs_margin_deg = 5. #smoothing area in degrees lat/lon
gw_wrf_margin_km = 10. #smoothing area in km
#gw_alpha = .143 #GW vertical interpolation by power law
gw_alpha = 1. #ignore wind power law, interpolate linearly


# chemical initial and boundary coords - CAMx nesting
# CAMx variable conversions
camx_conversions = {
        'NO':  dict(
            camx_vars = ['NO'],
            camx_units = ['ppmv'],
            output_unit = 'ppm',
            formula = lambda no, hlp: no,
            ),
        'NO2':  dict(
            camx_vars = ['NO2'],
            camx_units = ['ppmv'],
            output_unit = 'ppm',
            formula = lambda no2, hlp: no2,
            ),
        'NOX': dict(
            camx_vars=['NO', 'NO2'],
            camx_units=['ppmv', 'ppmv'],
            output_unit='ppm',
            formula=lambda no, no2, hlp: no + no2,
        ),
        'O3':  dict(
            camx_vars = ['O3'],
            camx_units = ['ppmv'],
            output_unit = 'ppm',
            formula = lambda o3, hlp: o3,
            ),
        'PM10':  dict(
            camx_vars = ['CPRM'],
            camx_units = ['micrograms m-3'],
            output_unit = 'kg/m3',
            formula = lambda cprm, hlp: (cprm+hlp.pm25) * 1e-9,
            ),
        'PM25':  dict(
            camx_vars = [],
            camx_units = [],
            output_unit = 'kg/m3',
            formula = lambda hlp: hlp.pm25 * 1e-9,
            ),
        }

camx_helpers = [
        ('pm25',  dict(
            camx_vars = 'PSO4 PNO3 PNH4 POA PEC FPRM SOA1 SOA2 SOA3 SOA4 SOPA SOPB'.split(),
            camx_units = ['micrograms m-3']*12,
            formula = lambda *args: np.sum(args[:-1], axis=0), #last arg is hlp
            )),
        ]


