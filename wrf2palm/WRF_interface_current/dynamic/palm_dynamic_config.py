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
'''Configuration module.

Configuration options are sourced into this module's globals.
'''

import os.path
from pathlib import Path
import inspect


# Just for PyCharm and similar IDEs to allow autocompletion from config values
if False:
    from palm_dynamic_defaults import *
    from palm_dynamic_init import *

def configure(configname):
    global dir_scripts
    # get path of the palm_dynamic script to source default config and init
    dir_scripts = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print('Running palm_dynamic from:', dir_scripts)
    # use config defaults
    configdefaultsfile = os.path.join(dir_scripts, "palm_dynamic_defaults.py")
    print('Default case config: ', configdefaultsfile)
    # user config file is located in configurations directory
    configfile = os.path.join(os.path.abspath(Path(dir_scripts).parent), "configurations", configname + '.conf')
    print('User case config:', configfile)
    # initialization of standard parameters done in script
    standardinitfile = os.path.join(dir_scripts, "palm_dynamic_init.py")
    print('Standard initialization: ', standardinitfile)
    # check existence of the supplied config file
    if not os.path.isfile(configfile):
        print("Config file " + configfile + " does not exists!")
        print_help()
        exit(2)
    # read default config values
    exec(open(configdefaultsfile).read(), globals())
    # read user configuration
    exec(open(configfile).read(), globals())
    # perform the standard initialization
    exec(open(standardinitfile).read(), globals())
