import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
sys.path.append('/scratch/ppcode/standard/sowfa_std')
import imp
import palm_data_ext
from palm_data_ext import *
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import numpy as np
from scipy.optimize import curve_fit
import funcs
from funcs import *
import matplotlib.pyplot as plt



""" SOWFA """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_0.0001_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# averaged coherence (ver)
freq, coh_av_0_d40h60, coh_std_0_d40h60, co_coh_av_0_d40h60, co_coh_std_0_d40h60, quad_coh_av_0_d40h60, quad_coh_std_0_d40h60, phase_av_0_d40h60, phase_std_0_d40h60 = coh_av_sowfa_z(0, 0, 50, 1, 3, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d80h60, coh_std_0_d80h60, co_coh_av_0_d80h60, co_coh_std_0_d80h60, quad_coh_av_0_d80h60, quad_coh_std_0_d80h60, phase_av_0_d80h60, phase_std_0_d80h60 = coh_av_sowfa_z(0, 0, 50, 0, 4, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d40h140, coh_std_0_d40h140, co_coh_av_0_d40h140, co_coh_std_0_d40h140, quad_coh_av_0_d40h140, quad_coh_std_0_d40h140, phase_av_0_d40h140, phase_std_0_d40h140 = coh_av_sowfa_z(0, 0, 50, 5, 7, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d80h140, coh_std_0_d80h140, co_coh_av_0_d80h140, co_coh_std_0_d80h140, quad_coh_av_0_d80h140, quad_coh_std_0_d80h140, phase_av_0_d80h140, phase_std_0_d80h140 = coh_av_sowfa_z(0, 0, 50, 4, 8, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10_refined'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1, coors_1 = PSD_data_sowfa(ppDir_1, 'prbg1', ((0,0,0),30.0), 'U', 0)
# averaged coherence (ver)
freq, coh_av_1_d40h60, coh_std_1_d40h60, co_coh_av_1_d40h60, co_coh_std_1_d40h60, quad_coh_av_1_d40h60, quad_coh_std_1_d40h60, phase_av_1_d40h60, phase_std_1_d40h60 = coh_av_sowfa_z(0, 0, 50, 1, 3, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d80h60, coh_std_1_d80h60, co_coh_av_1_d80h60, co_coh_std_1_d80h60, quad_coh_av_1_d80h60, quad_coh_std_1_d80h60, phase_av_1_d80h60, phase_std_1_d80h60 = coh_av_sowfa_z(0, 0, 50, 0, 4, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d40h140, coh_std_1_d40h140, co_coh_av_1_d40h140, co_coh_std_1_d40h140, quad_coh_av_1_d40h140, quad_coh_std_1_d40h140, phase_av_1_d40h140, phase_std_1_d40h140 = coh_av_sowfa_z(0, 0, 50, 5, 7, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d80h140, coh_std_1_d80h140, co_coh_av_1_d80h140, co_coh_std_1_d80h140, quad_coh_av_1_d80h140, quad_coh_std_1_d80h140, phase_av_1_d80h140, phase_std_1_d80h140 = coh_av_sowfa_z(0, 0, 50, 4, 8, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs10_0.01_refined'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, varSeq_2, coors_2 = PSD_data_sowfa(ppDir_2, 'prbg1', ((0,0,0),30.0), 'U', 0)
# averaged coherence (ver)
freq, coh_av_2_d40h60, coh_std_2_d40h60, co_coh_av_2_d40h60, co_coh_std_2_d40h60, quad_coh_av_2_d40h60, quad_coh_std_2_d40h60, phase_av_2_d40h60, phase_std_2_d40h60 = coh_av_sowfa_z(0, 0, 50, 1, 3, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d80h60, coh_std_2_d80h60, co_coh_av_2_d80h60, co_coh_std_2_d80h60, quad_coh_av_2_d80h60, quad_coh_std_2_d80h60, phase_av_2_d80h60, phase_std_2_d80h60 = coh_av_sowfa_z(0, 0, 50, 0, 4, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d40h140, coh_std_2_d40h140, co_coh_av_2_d40h140, co_coh_std_2_d40h140, quad_coh_av_2_d40h140, quad_coh_std_2_d40h140, phase_av_2_d40h140, phase_std_2_d40h140 = coh_av_sowfa_z(0, 0, 50, 5, 7, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d80h140, coh_std_2_d80h140, co_coh_av_2_d80h140, co_coh_std_2_d80h140, quad_coh_av_2_d80h140, quad_coh_std_2_d80h140, phase_av_2_d80h140, phase_std_2_d80h140 = coh_av_sowfa_z(0, 0, 50, 4, 8, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)



""" PALM """
prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_0.0001_main'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, varSeq_3 = PSD_data_palm(dir_3, jobName, 'M04', ['.001'], 'u')
# averaged coherence
freq, coh_av_3_d40h60, coh_std_3_d40h60, co_coh_av_3_d40h60, co_coh_std_3_d40h60, quad_coh_av_3_d40h60, quad_coh_std_3_d40h60, phase_av_3_d40h60, phase_std_3_d40h60 = coh_av_palm_z(0, 0, 50, 1, 3, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d80h60, coh_std_3_d80h60, co_coh_av_3_d80h60, co_coh_std_3_d80h60, quad_coh_av_3_d80h60, quad_coh_std_3_d80h60, phase_av_3_d80h60, phase_std_3_d80h60 = coh_av_palm_z(0, 0, 50, 0, 4, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d40h140, coh_std_3_d40h140, co_coh_av_3_d40h140, co_coh_std_3_d40h140, quad_coh_av_3_d40h140, quad_coh_std_3_d40h140, phase_av_3_d40h140, phase_std_3_d40h140 = coh_av_palm_z(0, 0, 50, 5, 7, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d80h140, coh_std_3_d80h140, co_coh_av_3_d80h140, co_coh_std_3_d80h140, quad_coh_av_3_d80h140, quad_coh_std_3_d80h140, phase_av_3_d80h140, phase_std_3_d80h140 = coh_av_palm_z(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)

prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, varSeq_4 = PSD_data_palm(dir_4, jobName, 'M04', ['.001'], 'u')
# averaged coherence
freq, coh_av_4_d40h60, coh_std_4_d40h60, co_coh_av_4_d40h60, co_coh_std_4_d40h60, quad_coh_av_4_d40h60, quad_coh_std_4_d40h60, phase_av_4_d40h60, phase_std_4_d40h60 = coh_av_palm_z(0, 0, 50, 1, 3, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d80h60, coh_std_4_d80h60, co_coh_av_4_d80h60, co_coh_std_4_d80h60, quad_coh_av_4_d80h60, quad_coh_std_4_d80h60, phase_av_4_d80h60, phase_std_4_d80h60 = coh_av_palm_z(0, 0, 50, 0, 4, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d40h140, coh_std_4_d40h140, co_coh_av_4_d40h140, co_coh_std_4_d40h140, quad_coh_av_4_d40h140, quad_coh_std_4_d40h140, phase_av_4_d40h140, phase_std_4_d40h140 = coh_av_palm_z(0, 0, 50, 5, 7, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d80h140, coh_std_4_d80h140, co_coh_av_4_d80h140, co_coh_std_4_d80h140, quad_coh_av_4_d80h140, quad_coh_std_4_d80h140, phase_av_4_d80h140, phase_std_4_d80h140 = coh_av_palm_z(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)

prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_0.01_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, varSeq_5 = PSD_data_palm(dir_5, jobName, 'M04', ['.000'], 'u')
# averaged coherence
freq, coh_av_5_d40h60, coh_std_5_d40h60, co_coh_av_5_d40h60, co_coh_std_5_d40h60, quad_coh_av_5_d40h60, quad_coh_std_5_d40h60, phase_av_5_d40h60, phase_std_5_d40h60 = coh_av_palm_z(0, 0, 50, 1, 3, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d80h60, coh_std_5_d80h60, co_coh_av_5_d80h60, co_coh_std_5_d80h60, quad_coh_av_5_d80h60, quad_coh_std_5_d80h60, phase_av_5_d80h60, phase_std_5_d80h60 = coh_av_palm_z(0, 0, 50, 0, 4, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d40h140, coh_std_5_d40h140, co_coh_av_5_d40h140, co_coh_std_5_d40h140, quad_coh_av_5_d40h140, quad_coh_std_5_d40h140, phase_av_5_d40h140, phase_std_5_d40h140 = coh_av_palm_z(0, 0, 50, 5, 7, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d80h140, coh_std_5_d80h140, co_coh_av_5_d80h140, co_coh_std_5_d80h140, quad_coh_av_5_d80h140, quad_coh_std_5_d80h140, phase_av_5_d80h140, phase_std_5_d80h140 = coh_av_palm_z(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)




""" 2*3 plots of coh_av """
# fitting
def fitting_func(f_, a):
    return np.exp(-a*f_)

fig = plt.figure()
fig.set_figwidth(15)
fig.set_figheight(8)
rNum, cNum = (2,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

## height = 60m
uz_0_60 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,2,varSeq_0) # sowfa
uz_3_60 = calc_uz_palm(2,varSeq_3) # palm
axs[0,0].plot(freq[1:]*40/uz_0_60, coh_av_0_d40h60[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_60, coh_av_0_d80h60[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_60, coh_av_3_d40h60[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_60, coh_av_3_d80h60[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[0,0].plot(freq[1:]*40/uz_0_60, coh_IEC(freq[1:], 40, uz_0_60, 8.1*42, 12, 00.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[0,0].plot(freq[1:]*80/uz_0_60, coh_IEC(freq[1:], 80, uz_0_60, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_0_60, coh_av_0_d40h60[1:25], bounds=(0, [100]))
axs[0,0].plot(freq*40/uz_0_60, fitting_func(freq*40/uz_0_60, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_0_60, coh_av_0_d80h60[1:25], bounds=(0, [100]))
axs[0,0].plot(freq*80/uz_0_60, fitting_func(freq*80/uz_0_60, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(0.0, 1.0)
for tick in axs[0,0].yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[0,0].set_ylabel(r"coh", fontsize=18)
axs[0,0].text(0.32, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,0].text(0.32, 0.6, r"$\mathrm{h = 60m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,0].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[0,0].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)

uz_1_60 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,2,varSeq_1) # sowfa
uz_4_60 = calc_uz_palm(2,varSeq_4) # palm
axs[0,1].plot(freq[1:]*40/uz_1_60, coh_av_1_d40h60[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_60, coh_av_1_d80h60[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_60, coh_av_4_d40h60[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_60, coh_av_4_d80h60[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[0,1].plot(freq[1:]*40/uz_1_60, coh_IEC(freq[1:], 40, uz_1_60, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[0,1].plot(freq[1:]*80/uz_1_60, coh_IEC(freq[1:], 80, uz_1_60, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_1_60, coh_av_1_d40h60[1:25], bounds=(0, [100]))
axs[0,1].plot(freq*40/uz_1_60, fitting_func(freq*40/uz_1_60, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_1_60, coh_av_1_d80h60[1:25], bounds=(0, [100]))
axs[0,1].plot(freq*80/uz_1_60, fitting_func(freq*80/uz_1_60, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(0.0, 1.0); axs[0,1].set_yticklabels([])
axs[0,1].text(0.32, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,1].text(0.32, 0.6, r"$\mathrm{h = 60m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,1].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[0,1].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)

uz_2_60 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,2,varSeq_2) # sowfa
uz_5_60 = calc_uz_palm(2,varSeq_5) # palm
axs[0,2].plot(freq[1:]*40/uz_2_60, coh_av_2_d40h60[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_60, coh_av_2_d80h60[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_60, coh_av_5_d40h60[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_60, coh_av_5_d80h60[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[0,2].plot(freq[1:]*40/uz_2_60, coh_IEC(freq[1:], 40, uz_2_60, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[0,2].plot(freq[1:]*80/uz_2_60, coh_IEC(freq[1:], 80, uz_2_60, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_2_60, coh_av_2_d40h60[1:25], bounds=(0, [100]))
axs[0,2].plot(freq*40/uz_2_60, fitting_func(freq*40/uz_2_60, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_2_60, coh_av_2_d80h60[1:25], bounds=(0, [100]))
axs[0,2].plot(freq*80/uz_2_60, fitting_func(freq*80/uz_2_60, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(0.0, 1.0); axs[0,2].set_yticklabels([])
axs[0,2].text(0.32, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,2].text(0.32, 0.6, r"$\mathrm{h = 60m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[0,2].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[0,2].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)


## height = 140m
uz_0_140 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_140 = calc_uz_palm(4,varSeq_3) # palm
axs[1,0].plot(freq[1:]*40/uz_0_140, coh_av_0_d40h140[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_140, coh_av_0_d80h140[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_140, coh_av_3_d40h140[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_140, coh_av_3_d80h140[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[1,0].plot(freq[1:]*40/uz_0_140, coh_IEC(freq[1:], 40, uz_0_140, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[1,0].plot(freq[1:]*80/uz_0_140, coh_IEC(freq[1:], 80, uz_0_140, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_0_140, coh_av_0_d40h140[1:25], bounds=(0, [100]))
axs[1,0].plot(freq*40/uz_0_140, fitting_func(freq*40/uz_0_140, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_0_140, coh_av_0_d80h140[1:25], bounds=(0, [100]))
axs[1,0].plot(freq*80/uz_0_140, fitting_func(freq*80/uz_0_140, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[1,0].set_xlim(0, 1.0)
for tick in axs[1,0].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[1,0].set_ylim(0.0, 1.0)
for tick in axs[1,0].yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[1,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=18)
axs[1,0].set_ylabel(r"coh", fontsize=18)
axs[1,0].text(0.32, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,0].text(0.32, 0.6, r"$\mathrm{h = 140m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,0].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[1,0].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)

uz_1_140 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_140 = calc_uz_palm(4,varSeq_4) # palm
axs[1,1].plot(freq[1:]*40/uz_1_140, coh_av_1_d40h140[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_140, coh_av_1_d80h140[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_140, coh_av_4_d40h140[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_140, coh_av_4_d80h140[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[1,1].plot(freq[1:]*40/uz_1_140, coh_IEC(freq[1:], 40, uz_1_140, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[1,1].plot(freq[1:]*80/uz_1_140, coh_IEC(freq[1:], 80, uz_1_140, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_1_140, coh_av_1_d40h140[1:25], bounds=(0, [100]))
axs[1,1].plot(freq*40/uz_1_140, fitting_func(freq*40/uz_1_140, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_1_140, coh_av_1_d80h140[1:25], bounds=(0, [100]))
axs[1,1].plot(freq*80/uz_1_140, fitting_func(freq*80/uz_1_140, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[1,1].set_xlim(0, 1.0)
for tick in axs[1,1].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[1,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=18)
axs[1,1].set_ylim(0.0, 1.0); axs[1,1].set_yticklabels([])
axs[1,1].text(0.32, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,1].text(0.32, 0.6, r"$\mathrm{h = 140m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,1].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[1,1].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)

uz_2_140 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_140 = calc_uz_palm(4,varSeq_5) # palm
axs[1,2].plot(freq[1:]*40/uz_2_140, coh_av_2_d40h140[1:], linestyle='', marker='o', markersize=5, color='r', mfc='none', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_140, coh_av_2_d80h140[1:], linestyle='', marker='x', markersize=5, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_140, coh_av_5_d40h140[1:], linestyle='', marker='o', markersize=5, color='b', mfc='none', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_140, coh_av_5_d80h140[1:], linestyle='', marker='x', markersize=5, color='b', label='palm-80m')
axs[1,2].plot(freq[1:]*40/uz_2_140, coh_IEC(freq[1:], 40, uz_2_140, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
axs[1,2].plot(freq[1:]*80/uz_2_140, coh_IEC(freq[1:], 80, uz_2_140, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')

popt_0, pcov = curve_fit(fitting_func, freq[1:25]*40/uz_2_140, coh_av_2_d40h140[1:25], bounds=(0, [100]))
axs[1,2].plot(freq*40/uz_2_140, fitting_func(freq*40/uz_2_140, *popt_0), linestyle='-.', color='k', label='Davenport-40m')
popt_1, pcov = curve_fit(fitting_func, freq[1:25]*80/uz_2_140, coh_av_2_d80h140[1:25], bounds=(0, [100]))
axs[1,2].plot(freq*80/uz_2_140, fitting_func(freq*80/uz_2_140, *popt_1), linestyle=':', color='k', label='Davenport-80m')

axs[1,2].set_xlim(0, 1.0)
for tick in axs[1,2].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[1,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=18)
axs[1,2].set_ylim(0.0, 1.0); axs[1,2].set_yticklabels([])
axs[1,2].text(0.32, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,2].text(0.32, 0.6, r"$\mathrm{h = 140m}$", fontsize=18) # transform=axs[0,0].transAxes
axs[1,2].text(0.32, 0.4, r'$\mathrm{a_{40m}=%5.2f}$' % tuple(popt_0), fontsize=18)
axs[1,2].text(0.32, 0.2, r'$\mathrm{a_{80m}=%5.2f}$' % tuple(popt_1), fontsize=18)

for i in range(2):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=18)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3,4,5,6,7]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,1.0), ncol=4, mode='None', borderaxespad=0, fontsize=18)
saveDir = '/scratch/projects/deepwind/photo/review'
saveName = 'fig8_coh_ver_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()