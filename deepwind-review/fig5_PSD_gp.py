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
import funcs
import matplotlib.pyplot as plt


segNum = 4096

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_0.0001_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, vSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_0, xSeq_0, ySeq_0, zSeq_0, wSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_0, PSD_u_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_u_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)
f_seq_0, PSD_v_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_v_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, vSeq_0, segNum)
f_seq_0, PSD_w_seq_0_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 0, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 4, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)
f_seq_0, PSD_w_seq_0_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_0, 8, xSeq_0.size, ySeq_0.size, wSeq_0, segNum)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10_refined'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1, coors_1 = PSD_data_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, vSeq_1, coors_1 = PSD_data_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_1, xSeq_1, ySeq_1, zSeq_1, wSeq_1, coors_1 = PSD_data_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_1, PSD_u_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_u_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)
f_seq_1, PSD_v_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_v_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, vSeq_1, segNum)
f_seq_1, PSD_w_seq_1_20 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 0, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_100 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 4, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)
f_seq_1, PSD_w_seq_1_180 = PSD_sowfa((144000.0, 151200.0, 0.5), tSeq_1, 8, xSeq_1.size, ySeq_1.size, wSeq_1, segNum)

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs10_0.01_refined'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = PSD_data_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, vSeq_2, coors_2 = PSD_data_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 1)
tSeq_2, xSeq_2, ySeq_2, zSeq_2, wSeq_2, coors_2 = PSD_data_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 2)
f_seq_2, PSD_u_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_u_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)
f_seq_2, PSD_v_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_v_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, vSeq_2, segNum)
f_seq_2, PSD_w_seq_2_20 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 0, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_100 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 4, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)
f_seq_2, PSD_w_seq_2_180 = PSD_sowfa((72000.0, 74400.0, 0.5), tSeq_2, 8, xSeq_2.size, ySeq_2.size, wSeq_2, segNum)




prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_0.0001_main'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3 = PSD_data_palm(dir_3, jobName, 'M03', ['.001'], 'u')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, vSeq_3 = PSD_data_palm(dir_3, jobName, 'M03', ['.001'], 'v')
tSeq_3, xSeq_3, ySeq_3, zSeq_3, wSeq_3 = PSD_data_palm(dir_3, jobName, 'M03', ['.001'], 'w')
f_seq_3, PSD_u_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_u_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)
f_seq_3, PSD_v_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_v_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, vSeq_3, segNum)
f_seq_3, PSD_w_seq_3_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 0, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 4, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)
f_seq_3, PSD_w_seq_3_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_3, 8, xSeq_3.size, ySeq_3.size, wSeq_3, segNum)

prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4 = PSD_data_palm(dir_4, jobName, 'M03', ['.001'], 'u')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, vSeq_4 = PSD_data_palm(dir_4, jobName, 'M03', ['.001'], 'v')
tSeq_4, xSeq_4, ySeq_4, zSeq_4, wSeq_4 = PSD_data_palm(dir_4, jobName, 'M03', ['.001'], 'w')
f_seq_4, PSD_u_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_u_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_u_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)
f_seq_4, PSD_v_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_v_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_v_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, vSeq_4, segNum)
f_seq_4, PSD_w_seq_4_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 0, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)
f_seq_4, PSD_w_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)
f_seq_4, PSD_w_seq_4_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 8, xSeq_4.size, ySeq_4.size, wSeq_4, segNum)

prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_0.01_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = PSD_data_palm(dir_5, jobName, 'M03', ['.000'], 'u')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, vSeq_5 = PSD_data_palm(dir_5, jobName, 'M03', ['.000'], 'v')
tSeq_5, xSeq_5, ySeq_5, zSeq_5, wSeq_5 = PSD_data_palm(dir_5, jobName, 'M03', ['.000'], 'w')
f_seq_5, PSD_u_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_u_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)
f_seq_5, PSD_v_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_v_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, vSeq_5, segNum)
f_seq_5, PSD_w_seq_5_20 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 0, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 4, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)
f_seq_5, PSD_w_seq_5_180 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_5, 8, xSeq_5.size, ySeq_5.size, wSeq_5, segNum)




""" 3*3 plots of scaled PSD (z0) """
segNum = 4096

fig = plt.figure()
fig.set_figwidth(12)
fig.set_figheight(10)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

colors_sowfa = plt.cm.Reds(np.linspace(0,1,6))
colors_sowfa = colors_sowfa[::-1]
colors_palm = plt.cm.Blues(np.linspace(0,1,6))
colors_palm = colors_palm[::-1]

# compute uz, uStar in SOWFA
uStar_0 = calc_uStar_sowfa(xSeq_0.size,ySeq_0.size,zSeq_0,uSeq_0,kappa=0.4,z0=0.0001)
uStar_1 = calc_uStar_sowfa(xSeq_1.size,ySeq_1.size,zSeq_1,uSeq_1,kappa=0.4,z0=0.001)
uStar_2 = calc_uStar_sowfa(xSeq_2.size,ySeq_2.size,zSeq_2,uSeq_2,kappa=0.4,z0=0.01)
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,0,uSeq_0)
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,uSeq_0)
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,8,uSeq_0)
uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,0,uSeq_1)
uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,uSeq_1)
uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,8,uSeq_1)
uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,0,uSeq_2)
uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,uSeq_2)
uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,8,uSeq_2)

# compute uz, uStar in PALM
uStar_3 = calc_uStar_palm(zSeq_3,uSeq_3,0,kappa=0.4,z0=0.0001)
uStar_4 = calc_uStar_palm(zSeq_4,uSeq_4,0,kappa=0.4,z0=0.001)
uStar_5 = calc_uStar_palm(zSeq_5,uSeq_5,0,kappa=0.4,z0=0.01)
uz_3_20 = calc_uz_palm(0,uSeq_3)
uz_3_100 = calc_uz_palm(4,uSeq_3)
uz_3_180 = calc_uz_palm(8,uSeq_3)
uz_4_20 = calc_uz_palm(0,uSeq_4)
uz_4_100 = calc_uz_palm(4,uSeq_4)
uz_4_180 = calc_uz_palm(8,uSeq_4)
uz_5_20 = calc_uz_palm(0,uSeq_5)
uz_5_100 = calc_uz_palm(4,uSeq_5)
uz_5_180 = calc_uz_palm(8,uSeq_5)

# kaimal spectrum
f_seq_kaimal = np.logspace(-4,2,2049)
# f_seq_kaimal = np.linspace(1e-4,1e0,2049)
kaimal_u_20_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_u_100_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_u_180_1 = funcs.kaimal_u(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
kaimal_v_20_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_v_100_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_v_180_1 = funcs.kaimal_v(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
kaimal_w_20_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_20, zSeq_1[0], uStar_1)
kaimal_w_100_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_100, zSeq_1[4], uStar_1)
kaimal_w_180_1 = funcs.kaimal_w(f_seq_kaimal, uz_1_180, zSeq_1[8], uStar_1)
f_seq_BP = np.logspace(-4,2,2049)
# f_seq_BP = np.linspace(1e-4,1e0,2049)
BP_w_20_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_20, zSeq_1[0], uStar_1)
BP_w_100_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_100, zSeq_1[4], uStar_1)
BP_w_180_1 = funcs.Busch_Panofsky_w(f_seq_BP, uz_1_180, zSeq_1[8], uStar_1)

axs[0,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_u_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,0].loglog(f_seq_0*20/uz_0_20, PSD_u_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,0].loglog(f_seq_1*20/uz_1_20, PSD_u_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,0].loglog(f_seq_2*20/uz_2_20, PSD_u_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,0].loglog(f_seq_3*20/uz_3_20, PSD_u_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,0].loglog(f_seq_4*20/uz_4_20, PSD_u_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,0].loglog(f_seq_5*20/uz_5_20, PSD_u_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,0].set_xlim(1e-2, 1e1); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(1e-4, 1e1)
for tick in axs[0,0].yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[0,0].set_ylabel(r"$\mathrm{fS_u/u^2_*}$", fontsize=18)
axs[0,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=18) # transform=axs[0,0].transAxes

axs[0,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_u_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,1].loglog(f_seq_0*100/uz_0_100, PSD_u_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,1].loglog(f_seq_1*100/uz_1_100, PSD_u_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,1].loglog(f_seq_2*100/uz_2_100, PSD_u_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,1].loglog(f_seq_3*100/uz_3_100, PSD_u_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,1].loglog(f_seq_4*100/uz_4_100, PSD_u_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,1].loglog(f_seq_5*100/uz_5_100, PSD_u_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,1].set_xlim(1e-2, 1e1); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(1e-4, 1e1); axs[0,1].set_yticklabels([])
axs[0,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=18) # transform=axs[0,1].transAxes

axs[0,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_u_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[0,2].loglog(f_seq_0*180/uz_0_180, PSD_u_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[0,2].loglog(f_seq_1*180/uz_1_180, PSD_u_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[0,2].loglog(f_seq_2*180/uz_2_180, PSD_u_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[0,2].loglog(f_seq_3*180/uz_3_180, PSD_u_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[0,2].loglog(f_seq_4*180/uz_4_180, PSD_u_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[0,2].loglog(f_seq_5*180/uz_5_180, PSD_u_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[0,2].set_xlim(1e-2, 1e1); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(1e-4, 1e1); axs[0,2].set_yticklabels([])
axs[0,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=18) # transform=axs[0,2].transAxes

axs[1,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_v_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,0].loglog(f_seq_0*20/uz_0_20, PSD_v_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,0].loglog(f_seq_1*20/uz_1_20, PSD_v_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,0].loglog(f_seq_2*20/uz_2_20, PSD_v_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,0].loglog(f_seq_3*20/uz_3_20, PSD_v_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,0].loglog(f_seq_4*20/uz_4_20, PSD_v_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,0].loglog(f_seq_5*20/uz_5_20, PSD_v_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,0].set_xlim(1e-2, 1e1); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(1e-4, 1e1)
for tick in axs[1,0].yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[1,0].set_ylabel(r"$\mathrm{fS_v/u^2_*}$", fontsize=18)
axs[1,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=18) # transform=axs[0,0].transAxes

axs[1,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_v_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,1].loglog(f_seq_0*100/uz_0_100, PSD_v_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,1].loglog(f_seq_1*100/uz_1_100, PSD_v_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,1].loglog(f_seq_2*100/uz_2_100, PSD_v_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,1].loglog(f_seq_3*100/uz_3_100, PSD_v_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,1].loglog(f_seq_4*100/uz_4_100, PSD_v_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,1].loglog(f_seq_5*100/uz_5_100, PSD_v_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,1].set_xlim(1e-2, 1e1); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(1e-4, 1e1); axs[1,1].set_yticklabels([])
axs[1,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=18) # transform=axs[1,1].transAxes

axs[1,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_v_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')

axs[1,2].loglog(f_seq_0*180/uz_0_180, PSD_v_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[1,2].loglog(f_seq_1*180/uz_1_180, PSD_v_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[1,2].loglog(f_seq_2*180/uz_2_180, PSD_v_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[1,2].loglog(f_seq_3*180/uz_3_180, PSD_v_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[1,2].loglog(f_seq_4*180/uz_4_180, PSD_v_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[1,2].loglog(f_seq_5*180/uz_5_180, PSD_v_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[1,2].set_xlim(1e-2, 1e1); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(1e-4, 1e1); axs[1,2].set_yticklabels([])
axs[1,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=18) # transform=axs[1,2].transAxes

axs[2,0].loglog(f_seq_kaimal*20/uz_1_20, kaimal_w_20_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,0].loglog(f_seq_BP*20/uz_1_20, BP_w_20_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,0].loglog(f_seq_0*20/uz_0_20, PSD_w_seq_0_20*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,0].loglog(f_seq_1*20/uz_1_20, PSD_w_seq_1_20*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,0].loglog(f_seq_2*20/uz_2_20, PSD_w_seq_2_20*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,0].loglog(f_seq_3*20/uz_3_20, PSD_w_seq_3_20*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,0].loglog(f_seq_4*20/uz_4_20, PSD_w_seq_4_20*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,0].loglog(f_seq_5*20/uz_5_20, PSD_w_seq_5_20*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,0].set_xlim(1e-2, 1e1)
axs[2,0].set_ylim(1e-4, 1e1)
for tick in axs[2,0].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
for tick in axs[2,0].yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[2,0].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=18)
axs[2,0].set_ylabel(r"$\mathrm{fS_w/u^2_*}$", fontsize=18)
axs[2,0].text(2e-2, 1e-3, r"$\mathrm{h = 20m}$", fontsize=18) # transform=axs[2,0].transAxes

axs[2,1].loglog(f_seq_kaimal*100/uz_1_100, kaimal_w_100_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,1].loglog(f_seq_BP*100/uz_1_100, BP_w_100_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,1].loglog(f_seq_0*100/uz_0_100, PSD_w_seq_0_100*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,1].loglog(f_seq_1*100/uz_1_100, PSD_w_seq_1_100*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,1].loglog(f_seq_2*100/uz_2_100, PSD_w_seq_2_100*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,1].loglog(f_seq_3*100/uz_3_100, PSD_w_seq_3_100*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,1].loglog(f_seq_4*100/uz_4_100, PSD_w_seq_4_100*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,1].loglog(f_seq_5*100/uz_5_100, PSD_w_seq_5_100*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,1].set_xlim(1e-2, 1e1)
axs[2,1].set_ylim(1e-4, 1e1); axs[2,1].set_yticklabels([])
for tick in axs[2,1].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[2,1].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=18)
axs[2,1].text(2e-2, 1e-3, r"$\mathrm{h = 100m}$", fontsize=18) # transform=axs[2,1].transAxes

axs[2,2].loglog(f_seq_kaimal*180/uz_1_180, kaimal_w_180_1*f_seq_kaimal/np.power(uStar_1,2), label='Kaimal', linewidth=1.0, linestyle='--', color='k')
axs[2,2].loglog(f_seq_BP*180/uz_1_180, BP_w_180_1*f_seq_BP/np.power(uStar_1,2), label='Busch-Panofsky', linewidth=1.0, linestyle=':', color='k')

axs[2,2].loglog(f_seq_0*180/uz_0_180, PSD_w_seq_0_180*f_seq_0/np.power(uStar_0,2), label='sowfa-0.0001m', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
axs[2,2].loglog(f_seq_1*180/uz_1_180, PSD_w_seq_1_180*f_seq_1/np.power(uStar_1,2), label='sowfa-0.001m', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
axs[2,2].loglog(f_seq_2*180/uz_2_180, PSD_w_seq_2_180*f_seq_2/np.power(uStar_2,2), label='sowfa-0.01m', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
axs[2,2].loglog(f_seq_3*180/uz_3_180, PSD_w_seq_3_180*f_seq_3/np.power(uStar_3,2), label='palm-0.0001m', linewidth=1.0, linestyle='-', color=colors_palm[0])
axs[2,2].loglog(f_seq_4*180/uz_4_180, PSD_w_seq_4_180*f_seq_4/np.power(uStar_4,2), label='palm-0.001m', linewidth=1.0, linestyle='-', color=colors_palm[1])
axs[2,2].loglog(f_seq_5*180/uz_5_180, PSD_w_seq_5_180*f_seq_5/np.power(uStar_5,2), label='palm-0.01m', linewidth=1.0, linestyle='-', color=colors_palm[2])
axs[2,2].set_xlim(1e-2, 1e1)
axs[2,2].set_ylim(1e-4, 1e1); axs[2,2].set_yticklabels([])
for tick in axs[2,2].xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
axs[2,2].set_xlabel(r"$\mathrm{fz/\overline{u}}$", fontsize=18)
axs[2,2].text(2e-2, 1e-3, r"$\mathrm{h = 180m}$", fontsize=18) # transform=axs[2,2].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=18)
handles, labels = axs[2,0].get_legend_handles_labels()
lgdord = [2,5,3,6,4,7,0,1]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=4, mode='None', borderaxespad=0, fontsize=16)
saveDir = '/scratch/projects/deepwind/photo/review'
saveName = 'fig5_PSD_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()