import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/sowfa/dataExt_L1')
import os
import matplotlib.pyplot as plt
import turbineOutputClass
from turbineOutputClass import *
import signalClass as sgn

''' 建立字典C准备储存各个算例的结果 '''
C = {}
caseName = {0:'NBL.ALM'}

''' case information '''
prjName = 'examples'
jobName = 'NBL'
prjDir = '/scratch/sowfadata/JOBS/' + prjName + '/'

for i in [0]:
    C[i] = Output(projDir_=prjDir, caseName_=caseName[i], nTurbine_=1, deltat_=0.1)
    C[i].P = C[i].powerRotor()
    C[i].T = C[i].thrust()


''' plot power curve '''
cn = 0
plt.figure(figsize = (8, 4))
x = C[cn].P[0][:,0]-18000
y = C[cn].P[0][:,1]/1e6
plt.plot(x, y, 'r-', linewidth = 1)
plt.ylabel('Power(MW)')
plt.xlabel('t(s)')
plt.title('Power Time Series')
plt.xlim(0,60)
plt.ylim(0,4)
plt.grid()
plt.show()

