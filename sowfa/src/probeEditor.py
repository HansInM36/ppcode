import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import numpy as np
import funcs

alpha = -30

zList = [0.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, \
         700.0, 750.0, 800.0, 900.0]
zNum = len(zList)


### along x-axis
xMin = 780.0
xMax = 1780.0
dx = 20.0

y = 1280.0

xNum = int((xMax - xMin) / dx + 1)
x = np.linspace(xMin, xMax, xNum)

f = open("/scratch/ppcode/sowfa/tmp/probesNx", "w")

for zInd in range(zNum):

    z = zList[zInd]

    coors = np.zeros((xNum,3))
    for i in range(xNum):
        coors[i][0], coors[i][1], coors[i][2] = x[i], y, z

    # probes's coordinate transmation
    O = ((xMin+xMax)/2,y,z)
    coors_ = funcs.trs(coors,O,alpha)
    for r in range(xNum):
        coors_[r] += O

    f.write('\n')
    f.write('    ' + 'probeNx' + str(zInd) + '\n')
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
    for i in range(xNum):
        f.write('                  ' + '(' + str(np.round(coors_[i,0]+0.01,2)) + ' ' + str(np.round(coors_[i,1]+0.01,2)) + ' ' + str(np.round(coors_[i,2]+0.01,2)) + ')' + '\n')
    f.write('          );' + '\n')
    f.write('    }' + '\n')

f.close()


### along y-axis
yMin = 780.0
yMax = 1780.0
dy = 20.0

x = 1280.0

yNum = int((yMax - yMin) / dx + 1)
y = np.linspace(yMin, yMax, yNum)


f = open("/scratch/ppcode/sowfa/tmp/probesNy", "w")

for zInd in range(zNum):

    z = zList[zInd]

    coors = np.zeros((yNum,3))
    for i in range(yNum):
        coors[i][0], coors[i][1], coors[i][2] = x, y[i], z

    # probes's coordinate transmation
    O = (x,(yMin+yMax)/2,z)
    coors_ = funcs.trs(coors,O,alpha)
    for r in range(xNum):
        coors_[r] += O

    f.write('\n')
    f.write('    ' + 'probeNy' + str(zInd) + '\n')
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
    for i in range(yNum):
        f.write('                  ' + '(' + str(np.round(coors_[i,0]+0.01,2)) + ' ' + str(np.round(coors_[i,1]+0.01,2)) + ' ' + str(np.round(coors_[i,2]+0.01,2)) + ')' + '\n')
    f.write('          );' + '\n')
    f.write('    }' + '\n')

f.close()



# yMin = 780.0
# yMax = 1780.0
# dy = 20.0
#
# zMin = 780.0
# zMax = 1780.0
# dz = 20.0
#
# yNum = int((yMax - yMin) / dy)
# zNum = int((zMax - zMin) / dz)
