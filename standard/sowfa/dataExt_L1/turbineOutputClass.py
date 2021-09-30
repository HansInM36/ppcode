import os
import csv
import numpy as np
from numpy import *
import pickle

''' functions tools '''
def strToNum_list(List):
    List = list(map(float, List))
    return List

def ave(Mat, t0=200, tn=320, col=1):
    " a function that computes the average of the outPut "
    row0 = int(np.where(Mat[:,0]==t0)[0])
    rown = int(np.where(Mat[:,0]==tn)[0])
    Mat = Mat[row0:rown,col]
    return np.mean(Mat)

class Output:
    # case information
    # projDir = 0 # directory of the project
    # caseName = 0 # name of the case
    # nTurbine = 0 # number of turbines
    # deltat = 0
    # P = 0
    # T = 0

    def __init__(self, projDir_, caseName_, nTurbine_, deltat_):
        " constructor "
        self.projDir = projDir_
        self.caseName = caseName_
        self.nTurbine = nTurbine_
        self.deltat = deltat_
        self.P = 0
        self.T = 0

    def powerRotor(self):
        " read powerRotor and return a dict "

        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        powerRotorDict_ = {}
        powerRotorDict = {}

        for startTime in startTimeList:
            file_powerRotor = open(self.projDir + self.caseName + '/turbineOutput/' + str(startTime) + '/powerRotor', 'r')
            data_powerRotor = csv.reader(file_powerRotor, delimiter=' ')
            rows_powerRotor = [row for row in data_powerRotor]
            powerRotorDict_[startTime] = dict(zip(turbines, turbines))
            for i in turbines:
                powerRotorDict_[startTime][i] = mat([[float(row[1]), float(row[3])] for row in rows_powerRotor[i+1::(self.nTurbine+1)]])

        for i in turbines: # 初始化一个0行矩阵，为了接下来可以在此基础上进行竖向拼接
            powerRotorDict[i] = np.zeros((0,powerRotorDict_[startTime][i].shape[1]))
            for startTime in startTimeList:
                startT = float(startTime)
                linkIndex = int((startT-float(startTimeList[0]))/self.deltat)
                powerRotorDict[i] = np.vstack((powerRotorDict[i][:linkIndex,:], powerRotorDict_[startTime][i]))
        return powerRotorDict

    def thrust(self):
        " read thrust and return a dict "

        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        thrustDict_ = {}
        thrustDict = {}

        for startTime in startTimeList:
            file_thrust = open(self.projDir + self.caseName + '/turbineOutput/' + str(startTime) + '/thrust', 'r')
            data_thrust = csv.reader(file_thrust, delimiter=' ')
            rows_thrust = [row for row in data_thrust]
            thrustDict_[startTime] = dict(zip(turbines, turbines))
            for i in turbines:
                thrustDict_[startTime][i] = mat([[float(row[1]), float(row[3])] for row in rows_thrust[i+1::(self.nTurbine+1)]])

        for i in turbines: # 初始化一个0行矩阵，为了接下来可以在此基础上进行竖向拼接
            thrustDict[i] = np.zeros((0,thrustDict_[startTime][i].shape[1]))
            for startTime in startTimeList:
                startT = float(startTime)
                linkIndex = int((startT-float(startTimeList[0]))/self.deltat)
                thrustDict[i] = np.vstack((thrustDict[i][:linkIndex,:], thrustDict_[startTime][i]))
        return thrustDict

    def torque(self):
        " read torque and return a dict "

        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        torqueDict_ = {}
        torqueDict = {}

        for startTime in startTimeList:
            file_torque = open(self.projDir + self.caseName + '/turbineOutput/' + str(startTime) + '/torqueRotor', 'r')
            data_torque = csv.reader(file_torque, delimiter=' ')
            rows_torque = [row for row in data_torque]
            torqueDict_[startTime] = dict(zip(turbines, turbines))
            for i in turbines:
                torqueDict_[startTime][i] = mat([[float(row[1]), float(row[3])] for row in rows_torque[i+1::(self.nTurbine+1)]])

        for i in turbines: # 初始化一个0行矩阵，为了接下来可以在此基础上进行竖向拼接
            torqueDict[i] = np.zeros((0,torqueDict_[startTime][i].shape[1]))
            for startTime in startTimeList:
                startT = float(startTime)
                linkIndex = int((startT-float(startTimeList[0]))/self.deltat)
                torqueDict[i] = np.vstack((torqueDict[i][:linkIndex,:], torqueDict_[startTime][i]))
        return torqueDict

    def azimuth(self):
        " read azimuth and return a dict "

        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        azimuthDict_ = {}
        azimuthDict = {}

        for startTime in startTimeList:
            file_azimuth = open(self.projDir + self.caseName + '/turbineOutput/' + str(startTime) + '/azimuth', 'r')
            data_azimuth = csv.reader(file_azimuth, delimiter=' ')
            rows_azimuth = [row for row in data_azimuth]
            azimuthDict_[startTime] = dict(zip(turbines, turbines))
            for i in turbines:
                azimuthDict_[startTime][i] = mat([[float(row[1]), float(row[3])] for row in rows_azimuth[i+1::(self.nTurbine+1)]])

        for i in turbines: # 初始化一个0行矩阵，为了接下来可以在此基础上进行竖向拼接
            azimuthDict[i] = np.zeros((0,azimuthDict_[startTime][i].shape[1]))
            for startTime in startTimeList:
                startT = float(startTime)
                linkIndex = int((startT-float(startTimeList[0]))/self.deltat)
                azimuthDict[i] = np.vstack((azimuthDict[i][:linkIndex,:], azimuthDict_[startTime][i]))
        return azimuthDict

    def axialForce(self, segNum_):
        " read azimuth and return a dict "

        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        axialForceDict_ = {}
        axialForceDict = {}

        for startTime in startTimeList:
            file_axialForce = open(self.projDir + self.caseName + '/turbineOutput/' + str(startTime) + '/axialForce', 'r')
            data_axialForce = csv.reader(file_axialForce, delimiter=' ')
            rows_axialForce = [row for row in data_axialForce]
            axialForceDict_[startTime] = dict(zip(turbines, turbines))
            for i in turbines:
                # axialForceDict_[startTime][i] = mat([[float(row[1]), float(row[3])] for row in rows_axialForce[i+1::(self.nTurbine+1)]])
                axialForceDict_[startTime][i] = {}
                for j in range(3):
                    axialForceDict_[startTime][i][j] = array([[float(row[k+4]) for k in range(segNum_)] for row in rows_axialForce[i*3+j+1::(self.nTurbine*3+1)]])

        for i in turbines: # 初始化一个0行矩阵，为了接下来可以在此基础上进行竖向拼接
            # axialForceDict[i] = np.zeros((0,azimuthDict_[startTime][i].shape[1]))
            axialForceDict[i] = {}
            for j in range(3):
                axialForceDict[i][j] = np.zeros((0,axialForceDict_[startTime][i][j].shape[1]))
            for startTime in startTimeList:
                startT = float(startTime)
                linkIndex = int((startT-float(startTimeList[0]))/self.deltat)
                for j in range(3):
                    axialForceDict[i][j] = np.vstack((axialForceDict[i][j][:linkIndex,:], axialForceDict_[startTime][i][j]))
        return axialForceDict

    def Moop(self, radii_ ,segNum_):
        startTimeList = os.listdir(self.projDir + self.caseName + '/turbineOutput/' + '.')
        startTimeList.sort()

        turbines = list(range(0,self.nTurbine))

        xFDict = self.axialForce(segNum_)
        aziDict = self.azimuth()

        MoopDict = {}

        for i in turbines:
            MoopDict[i] = {}
            for j in range(3):
                MoopDict[i][j] = np.zeros((aziDict[0].shape[0],2))

        for i in turbines:
            for j in range(3):
                for t in range(aziDict[0].shape[0]):
                    MoopDict[i][j][t,0] = aziDict[i][t,0]
                    Msum = 0
                    for n in range(segNum_):
                        Msum += radii_/segNum_*(n+0.5) * xFDict[i][j][t,n]
                    MoopDict[i][j][t,1] = Msum
        return MoopDict




    def Myaw(self, radii_, segNum_):
        xFDict = self.axialForce(segNum_)
        aziDict = self.azimuth()

        turbines = list(range(0,self.nTurbine))

        MyawDict = {}
        for i in turbines:
            MyawDict[i] = np.zeros((aziDict[0].shape[0],2))

        for i in turbines:
            for j in range(aziDict[0].shape[0]):
                MyawDict[i][j,0] = aziDict[i][j,0]
                Msum = 0
                for k in range(3):
                    for n in range(segNum_):
                        Msum += xFDict[i][k][j,n] * np.sin((aziDict[i][j,1]+k*120)*np.pi/180) * (radii_/segNum_*(n+0.5))
                MyawDict[i][j,1] = Msum
        return MyawDict
