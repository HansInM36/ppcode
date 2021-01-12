import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
import pickle
import funcs

xList = []
yList = []
zList = []

""" input """
# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

prbgName = 'prbg2'

# coordinate transformation
O = (1280, 1280, 0)
alpha = -30

# input min, max and d for x, y, and z
xMin, xMax, dx = (1280, 1280, 0)
yMin, yMax, dy = (1280, 1280, 0)
zMin, zMax, dz = (20, 180, 20)

# or directly input the list of x, y, z
zList = list(np.arange(20,200,20)) + [200.0, 400.0, 600.0]

if len(xList) == 0:
    if dx != 0:
        I = int((xMax - xMin) / dx + 1)
        xList = list(np.linspace(xMin, xMax, I))
    else: xList = [xMin]; I = 1
else: I = len(xList)

if len(yList) == 0:
    if dy != 0:
        J = int((yMax - yMin) / dy + 1)
        yList = list(np.linspace(yMin, yMax, J))
    else: yList = [yMin]; J = 1
else: J = len(yList)

if len(zList) == 0:
    if dz != 0:
        K = int((zMax - zMin) / dz + 1)
        zList = list(np.linspace(zMin, zMax, K))
    else: zList = [zMin]; K = 1
else: K = len(zList)

''' save (I,J,K) into a binary file with pickle '''
f = open(ppDir + '/data/' + prbgName + '_info', 'wb')
pickle.dump([O, alpha, xList, yList, zList], f)
f.close()

coorList = []
for k in range(K):
    for j in range(J):
        for i in range(I):
            coorList.append((xList[i], yList[j], zList[k]))

coors = np.array(coorList)
coors = funcs.trs(coors, O, alpha)
for p in range(I*J*K):
    coors[p] += O


f = open(ppDir + '/data/' + prbgName + '_dict', "w")
f.write('\n')
f.write('    ' + prbgName + '\n')
f.write('    {' + '\n')
f.write('          type                probes;' + '\n')
f.write('          functionObjectLibs ("libsampling.so");' + '\n')
f.write('          outputControl       runTime;' + '\n')
f.write('          writeInterval       0.1;' + '\n')
f.write('          timeStart           0.0;' + '\n')
f.write('          fields' + '\n')
f.write('          (' + '\n')
f.write('              U' + '\n')
f.write('          );' + '\n')
f.write('' + '\n')
f.write('          probeLocations' + '\n')
f.write('          (' + '\n')
for p in range(I*J*K):
    f.write('                  ' + '(' + str(np.round(coors[p,0]+0.01,2)) + ' ' + str(np.round(coors[p,1]+0.01,2)) + ' ' + str(np.round(coors[p,2]+0.01,2)) + ')' + '\n')
f.write('          );' + '\n')
f.write('    }' + '\n')
f.close()
