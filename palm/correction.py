#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:54:11 2019

@author: miw011


Run as

python correction.py U D ug_surface vg_surface ug_output vg_output


Needs six variables from user:

U - the desired wind speed in x-direction at the hub height, [m/s]
(The flow is aligned with x-axis, so V = 0 m/s)

D - flow direction relative to positive x-direction, [degrees]

Values used as INPUT PARAMETERS for the simulation run:
ug_surface - u-component of the geostrophic wind, [m/s]
vg_surface - v-component of the geostrophic wind, [m/s]

Values as CALCULATED by the simulation run in the steady state:
ug_output - u-component of the flow at the hub height or at the closest grid point, [m/s]
vg_output - v-component of the flow at the hub height or at the closest grid point, [m/s]


The script will then return two values to set as new ug_surface and vg_surface.
Use RECOMMENDED VALUES from the script output.

"""


import numpy as np
from math import pi

import sys

#%%
if len(sys.argv)<2:
    print('Not enough parameters were provided to calculate the correction.')
    print('Please provide six parameters: \n\
          the desired wind speed and direction, \n\
          u and v as in the input file, \n\
          u and v as returned by the simulation at the hub height.\n\
          Make sure that the simulation had reached the steady state, or the correction may be insufficient.')

else:
    ## desired values
    U = float(sys.argv[1]) if len(sys.argv)>1 else 15.0
    D = float(sys.argv[2]) if len(sys.argv)>2 else 0.0

    ## input values
    ug_surface = float(sys.argv[3]) if len(sys.argv)>3 else 15.0
    vg_surface = float(sys.argv[4]) if len(sys.argv)>4 else 0.0

    ## output values
    ug_output = float(sys.argv[5]) if len(sys.argv)>5 else 15.0
    vg_output = float(sys.argv[6]) if len(sys.argv)>6 else 0.0

    # convert direction to 360 degree system where 270 degrees = west direction
    d=D/360*2*pi

    U_input = np.sqrt(ug_surface**2 + vg_surface**2)
    d_input = np.arcsin(vg_surface/U_input)

    U_output = np.sqrt(ug_output**2 + vg_output**2)
    d_output = np.arcsin(vg_output/U_output)

    ## correction
    dd = d-d_output
    ku = U/U_output

    d_corr = d_input+dd
    U_corr = ku*U_input

    ug_corr=U_corr*np.cos(d_corr)
    vg_corr=U_corr*np.sin(d_corr)

    #%%
    print('INPUT VALUES')
    print('ug = {:6.3f} m/s'.format(ug_surface))
    print('vg = {:6.3f} m/s'.format(vg_surface))
    print('\n')
    print('OUTPUT VALUES')
    print('ug = {:6.3f} m/s'.format(ug_output))
    print('vg = {:6.3f} m/s'.format(vg_output))
    print('\n')
    print('CORRECTION')
    print('rotate vector by {:6.3f} rad'.format(dd))
    print('extend vector by the factor of {:6.3f}'.format(ku))
    print('\n')
    print('RECOMMENDED VALUES')
    print('ug = {:6.3f} m/s'.format(ug_corr))
    print('vg = {:6.3f} m/s'.format(vg_corr))
