import numpy as np


### along x-axis
xMin = 780.0
xMax = 1780.0
dx = 20.0

y = 1280.0

zList = [0.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, \
         700.0, 750.0, 800.0, 900.0]
zNum = len(zList)

xNum = int((xMax - xMin) / dx + 1)
x = np.linspace(xMin, xMax, xNum)


f = open("/scratch/ppcode/sowfa/tmp/probesNx", "w")

for zInd in range(zNum):
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
        f.write('                  ' + '(' + str(np.round(x[i]+0.01,2)) + ' ' + str(np.round(y+0.01,2)) + ' ' + str(np.round(zList[zInd]+0.01,2)) + ')' + '\n')
    f.write('          );' + '\n')
    f.write('    }' + '\n')

f.close()


### along y-axis
yMin = 780.0
yMax = 1780.0
dy = 20.0

x = 1280.0

zList = [0.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, \
         700.0, 750.0, 800.0, 900.0]
zNum = len(zList)

yNum = int((yMax - yMin) / dx + 1)
y = np.linspace(yMin, yMax, yNum)


f = open("/scratch/ppcode/sowfa/tmp/probesNy", "w")

for zInd in range(zNum):
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
        f.write('                  ' + '(' + str(np.round(x+0.01,2)) + ' ' + str(np.round(y[i]+0.01,2)) + ' ' + str(np.round(zList[zInd]+0.01,2)) + ')' + '\n')
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
