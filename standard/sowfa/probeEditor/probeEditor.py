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
prjName = 'examples'
jobName = 'NBL'
jobDir = '/scratch/sowfadata/JOBS/' + prjName + '/' + jobName

prbgName = 'prbg0'

# rotate the line of probes (O is the axis of rotation)
O = (640, 640, 0)
alpha = -30 # minus for rotate in counterclockwise (degree)

# input min, max and d for x, y, and z (in original coordinate)
xMin, xMax, dx = (320, 960, 80)
yMin, yMax, dy = (640, 640, 0)
zMin, zMax, dz = (10, 100, 10)

# or directly input the list of x, y, z
# zList = list(np.arange(20,200,20)) + [200.0, 400.0, 600.0]
# zList = [20,100,180]

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
f = open(jobDir + '/data/' + prbgName + '_info', 'wb')
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


f = open(jobDir + '/data/' + prbgName + '_dict', "w")
f.write('\n')
f.write('    ' + prbgName + '\n')
f.write('    {' + '\n')
f.write('          type                probes;' + '\n')
f.write('          functionObjectLibs ("libsampling.so");' + '\n')
f.write('          interpolationScheme cellPointFace;' + '\n')
f.write('          outputControl       runTime;' + '\n')
f.write('          writeInterval       0.5;' + '\n')
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
