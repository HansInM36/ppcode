import numpy as np
from numpy import *

# 定义一个函数使用aveData计算平均速度
def Umean_(Dict, alpha):
    a = np.cos(np.pi/180 * (270-alpha))
    b = np.sin(np.pi/180 * (270-alpha))

    itemList = list(Dict.keys())
    itemList.remove('H')

    timeList = list(Dict[itemList[0]])
    timeN = len(timeList)

    Mshape = Dict[itemList[0]][timeList[0]].shape

    Umean = np.zeros(Mshape)
    for time in timeList:
        Umean += Dict['U_mean'][time] * a + Dict['V_mean'][time] * b
    Umean /= timeN
    return Umean
# 定义一个函数使用aveData计算湍流强度
def TI_(Dict, alpha):
    a = np.cos(np.pi/180 * (270-alpha))
    b = np.sin(np.pi/180 * (270-alpha))

    itemList = list(Dict.keys())
    itemList.remove('H')

    timeList = list(Dict[itemList[0]])
    timeN = len(timeList)

    Mshape = Dict[itemList[0]][timeList[0]].shape

    TIx = np.zeros(Mshape)
    TIy = np.zeros(Mshape)
    TIz = np.zeros(Mshape)

    uu = np.zeros(Mshape)
    vv = np.zeros(Mshape)
    Umean = np.zeros(Mshape)
    for time in timeList:
        Umean += Dict['U_mean'][time] * a + Dict['V_mean'][time] * b
        uu = Dict['uu_mean'][time] * a**2 + Dict['vv_mean'][time] * b**2 + 2 * Dict['uv_mean'][time] * a*b
        vv = Dict['uu_mean'][time] * b**2 + Dict['vv_mean'][time] * a**2 - 2 * Dict['uv_mean'][time] * a*b
        ww = Dict['vv_mean'][time]
        TIx += uu
        TIy += vv
        TIz += ww

    Umean /= timeN
    TIx = np.power(TIx/timeN,0.5) / Umean
    TIy = np.power(TIy/timeN,0.5) / Umean
    TIz = np.power(TIz/timeN,0.5) / Umean

    return TIx, TIy, TIz

def flt_seq(x,tao):
    """ 以tao来过滤一个序列 """
    '''
    filter x with tao
    x must be a 1D array
    '''
    '''
    x = np.array([1,2,3,4,5,6,7])
    flt_seq(x,3)
    '''
    tao = int(tao)
    l = np.shape(x)[0]
    y = np.zeros(np.shape(x))

    for i in range(l):
        if i-tao < 0:
            a = 0
        else:
            a = i-tao
        if i+tao+1 > l:
            b = l
        else:
            b = i+tao+1
        a, b = int(a), int(b)
        y[i] = sum(x[a:b]) / np.shape(x[a:b])[0]
    return y


class Ave:
    """ members """

    ''' constructor '''
    def __init__(self, avedata, alpha):
        """ 初始化参数为一个储存了averaging data的字典， 一级key是'H'以及'U_mean','V_mean'等物理量，二级key是时刻, 和一个入流方向（地理坐标系） """
        self.aveData = avedata
        self.wd = (270 - alpha) * np.pi/180 # 这里已经转换为数学坐标系并以弧度为单位

        self.itemList = list(avedata.keys())
        self.itemList.remove('H') # 除去'H'

        self.timeList = list(avedata[self.itemList[0]].keys())
        self.timeList.sort()
        self.tN = len(self.timeList)

        self.Mshape = avedata[self.itemList[0]][self.timeList[0]].shape

    def ws_(self):
        ''' 该函数根据aveData计算时均速度廓线 '''
        a = np.cos(self.wd)
        b = np.sin(self.wd)

        self.wsp = np.zeros(self.Mshape)

        for time in self.timeList:
            self.wsp += self.aveData['U_mean'][time] * a + self.aveData['V_mean'][time] * b
        self.wsp /= self.tN
        self.wsp = np.hstack((self.aveData['H'],self.wsp))
        return self.wsp

    def TI_(self):
        ''' 该函数根据aveData计算湍流强度廓线 '''
        a = np.cos(self.wd)
        b = np.sin(self.wd)

        TIx = np.zeros(self.Mshape)
        TIy = np.zeros(self.Mshape)
        TIz = np.zeros(self.Mshape)

        uu = np.zeros(self.Mshape)
        vv = np.zeros(self.Mshape)
        Umean = np.zeros(self.Mshape)

        for time in self.timeList:
            Umean += self.aveData['U_mean'][time] * a + self.aveData['V_mean'][time] * b
            uu = self.aveData['uu_mean'][time] * a**2 + self.aveData['vv_mean'][time] * b**2 + 2 * self.aveData['uv_mean'][time] * a*b
            vv = self.aveData['uu_mean'][time] * b**2 + self.aveData['vv_mean'][time] * a**2 - 2 * self.aveData['uv_mean'][time] * a*b
            ww = self.aveData['vv_mean'][time]
            TIx += uu
            TIy += vv
            TIz += ww

        Umean /= self.tN
        TIx = np.power(TIx/self.tN,0.5) / Umean
        TIy = np.power(TIy/self.tN,0.5) / Umean
        TIz = np.power(TIz/self.tN,0.5) / Umean

        return TIx, TIy, TIz


class Wake:
    """ members """
    wakeData = 0 # wake data dict
    timeList = 0
    startTime = 0
    stopTime = 0
    tNum = 0

    secList = 0

    ''' constructor '''
    def __init__(self, wakeDataDict):
        self.wakeData = wakeDataDict

        self.timeList = list(wakeDataDict.keys())
        self.timeList.sort()
        self.startTime = min([float(i) for i in self.timeList])
        self.stopTime = max([float(i) for i in self.timeList])
        self.tNum = len(self.timeList)

        self.secList = list(wakeDataDict[self.timeList[0]].keys())
        self.secList.sort()

    ''' functions '''
    def ave_wakeData(self):
        """ this function compute the time average velocities of each wake section """
        wakeDataDict_ave = dict(zip(self.secList, self.secList))
        for i in self.secList:
            secSum = mat(zeros(shape(self.wakeData[self.timeList[0]][i][:,4:7]))) # initialize a mat for summing data of sec i in all times
            for j in self.timeList:
                secSum += self.wakeData[j][i][:,4:7]
            wakeDataDict_ave[i] = secSum / self.tNum
            wakeDataDict_ave[i] = c_[self.wakeData[self.timeList[0]][i][:,0:4], wakeDataDict_ave[i]]
        return wakeDataDict_ave #返回一个字典，key为截面，value是7列矩阵

    def intensity(self):
        """ this function compute the turbulence intensity of Vx, Vy, Vz of each wake section """
        ave = self.ave_wakeData()
        intensity = dict(zip(self.secList, self.secList))
        for section in self.secList:
            v_ave = ave[section][:,4:7] + 0.0001 # 加一个小量避免出现分母为0
            sum2 = np.zeros(v_ave.shape)
            for time in self.timeList:
                v = self.wakeData[time][section][:,4:7]
                sum2 = sum2 + np.power((v - v_ave), 2)
            sum2 = sum2 / len(self.timeList)
            sum2 = np.power(sum2, 0.5)
            intensity[section] = c_[self.wakeData[self.timeList[0]][section][:,0:4], (sum2 / v_ave[:,0])]
        return intensity #返回一个字典，key为截面，value是7列矩阵

    def ReStr(self, D):
        """ this function compute the Reynold Stress of each wake section """
        '''
        D is a tuple, 比如(0,2) 0代表x，1代表y，2代表z
        '''
        ave = self.ave_wakeData()
        RS = dict(zip(self.secList, self.secList))
        for section in self.secList:
            v_ave = ave[section][:,4:7] + 0.0001 # 加一个小量避免出现分母为0
            sum2 = np.zeros((v_ave.shape[0],1))
            for time in self.timeList:
                v = self.wakeData[time][section][:,4:7]
                tv = v - v_ave
                sum2 = sum2 + np.multiply(tv[:,D[0]], tv[:,D[1]])
            sum2 = sum2 / len(self.timeList)
            RS[section] = c_[self.wakeData[self.timeList[0]][section][:,0:4], sum2]
            RS[section] = c_[RS[section], np.zeros((v_ave.shape[0],2))] # 这里是小瑕疵，因为网格化插值必须要求输入是7列矩阵，所以补上两列0向量，没有实际意义
        return RS

    def secExt(self, sec):
        """ this function extract the wake data of a certain section, returning a dict of which the keys are times the values are n*7 mat """
        '''
        sec = 'Sec6D'
        '''
        dict = {}
        for time in self.timeList:
            dict[time] = self.wakeData[time][sec]
        return dict



class Sec(object):
    """ 初始化参数为一个储存了某一截面尾流信息的字典，key值是时刻，键值是pNum行7列的array """

    secData = {}
    topoData = 0 # 储存点编号，坐标的信息
    timeList = []
    pNum = 0
    tNum = 0
    secData_fd = {} # 储存经过tao过滤后的截面信息，字典形式和secData一样
    tseq = 0

    def __init__(self, secDataDict):
        self.secData = secDataDict
        self.timeList = list(secDataDict.keys())
        self.timeList.sort()
        self.topoData = secDataDict[self.timeList[0]][:,0:4]
        self.tNum = len(self.timeList)
        self.pNum = shape(secDataDict[self.timeList[0]])[0]
        self.tseq = array(self.timeList)

    def V_tSeq(self, r, axis):
        """ 抽取第secData第r行axis轴对应速度的时间序列 """
        '''
        r是行数，是一个整数，第一行r=0
        axis是str，'x','y'或'z'
        '''
        vseq = array(zeros((shape(self.tseq))))
        if axis == 'x':
            for i in range(self.tNum):
                vseq[i] = self.secData[self.timeList[i]][r,4]
        elif axis == 'y':
            for i in range(self.tNum):
                vseq[i] = self.secData[self.timeList[i]][r,5]
        elif axis == 'z':
            for i in range(self.tNum):
                vseq[i] = self.secData[self.timeList[i]][r,6]
        else:
            return print('wrong axis')
        return vseq

    def fSec(self, tao):
        """ 用tao过滤整个截面的信息，得到self.secData_fd """
        for t in range(self.tNum):
            self.secData_fd[self.timeList[t]] = array(zeros((self.pNum,3)))
        for r in range(self.pNum):
            Vx = self.V_tSeq(r,'x')
            Vy = self.V_tSeq(r,'y')
            Vz = self.V_tSeq(r,'z')
            Vx = flt_seq(Vx, tao)
            Vy = flt_seq(Vy, tao)
            Vz = flt_seq(Vz, tao)
            for t in range(self.tNum):
                self.secData_fd[self.timeList[t]][r,0] = Vx[t]
                self.secData_fd[self.timeList[t]][r,1] = Vy[t]
                self.secData_fd[self.timeList[t]][r,2] = Vz[t]
        for t in range(self.tNum):
            self.secData_fd[self.timeList[t]] = hstack((self.topoData, self.secData_fd[self.timeList[t]]))

    def fSec_t(self, tao, t_str):
        """ 用tao过滤整个截面的信息，得到t_str时刻的过滤后的截面的信息 """
        '''
        t_str必须是一个包含于self.timeList里面的某个str
        '''
        sec_t_fd = array(zeros((self.pNum,3)))

        tDict = {}
        for tt in range(self.tNum): # 为简化计算，建立一个序数和时刻str一一对应的字典
            tDict[tt] = self.timeList[tt]
            if tDict[tt] == t_str:
                t = tt # t_str则对应于序数t
        if t-tao < 0:
            a = 0
        else:
            a = t-tao
        if t+tao+1 > self.tNum:
            b = self.tNum
        else:
            b = t+tao+1
        a, b = int(a), int(b)

        for r in range(self.pNum):
            Vx = 0
            Vy = 0
            Vz = 0
            for i in range(a,b):
                Vx += self.secData[tDict[i]][r,4]
                Vy += self.secData[tDict[i]][r,5]
                Vz += self.secData[tDict[i]][r,6]
            Vx = Vx/len(range(a,b))
            Vy = Vy/len(range(a,b))
            Vz = Vz/len(range(a,b))
            sec_t_fd[r,0] = Vx
            sec_t_fd[r,1] = Vy
            sec_t_fd[r,2] = Vz
        return hstack((self.topoData, sec_t_fd))

    def fSec_p(self, tao, p):
        """ 用tao过滤某一点的时间序列，得到一个过滤后某点的时间序列 """
        '''
        p 是一个tuple，储存三个维度的坐标，比如（1000,300,450）
        '''
        data = self.secData
        timeList = self.timeList
        # 找 data 里面与目标点距离最近的点(r 就是该点的行号)，并以该点进行过滤
        min = 99999
        for r in self.topoData:
            temp = (r[0,1]-p[0])**2 + (r[0,2]-p[1])**2 + (r[0,3]-p[2])**2
            if temp < min:
                min = temp
                rmin = int(r[0,0]-1) # 行号和点号差了1

        Vx = self.V_tSeq(rmin,'x')
        Vy = self.V_tSeq(rmin,'y')
        Vz = self.V_tSeq(rmin,'z')
        return vstack((Vx,Vy,Vz)).T


class SecITP:
    ''' 主要用于对某个平面信息进行网格化插值 '''
    """ members """
    secData = 0
    meshData = 0
    meshData_horizAve = 0

    ''' constructor '''
    def __init__(self, wakeMat): # constructor
        self.secData = wakeMat # wakeMat is a mat of pNum*7 (the first column is the pointID)

    ''' functions '''
    def get_wakeData(self, wakeMat): # function for reassigning
        self.secData = wakeMat

    def segmentSum(self, inMat, segNum, uNumOfSeg):
        """ this function is for the compute of horizontal averaging mesh data """
        outMat = mat(zeros((segNum,1)))
        for i in range(0,segNum):
            outMat[i,0] = sum(inMat[i*uNumOfSeg:(i+1)*uNumOfSeg,0])
        outMat = outMat / uNumOfSeg
        return outMat

    def meshITP_Ny(self,x_axis,z_axis): # interpolate the original wake data and project it onto a structral mesh
        """ x_axis should be a tuple indicating the min, max value and the number of segments, like x_axis = (-240, 240, 60), and similarly the z_axis """
        pNum_x = x_axis[2] + 1
        pNum_z = z_axis[2] + 1
        delta_x = (x_axis[1] - x_axis[0])/x_axis[2]
        delta_z = (z_axis[1] - z_axis[0])/z_axis[2]
        coor_x = range(int(x_axis[0]), int(x_axis[1] + delta_x), int(delta_x))
        coor_z = range(int(z_axis[0]), int(z_axis[1] + delta_z), int(delta_z))
        coor_y = sum(self.secData[:,2])/shape(self.secData[:,2])[0] # all y coordinate are the same, because this is a section of yNormal
        coors_org = self.secData[:,1:4] # build the original points mat

        # compute mesh coordinates and velocities
        from scipy.interpolate import griddata
        [Xmesh, Zmesh] = meshgrid(coor_x, coor_z)
        Ymesh = mat(ones((shape(Xmesh)))) * coor_y
        Xmesh = Xmesh.ravel()
        Ymesh = Ymesh.ravel()
        Zmesh = Zmesh.ravel()
        coors_mesh = (vstack((Xmesh,Ymesh,Zmesh))).T # build the mesh points mat

        Vx = griddata(coors_org, self.secData[:,4], coors_mesh, method = 'nearest')
        Vy = griddata(coors_org, self.secData[:,5], coors_mesh, method = 'nearest')
        Vz = griddata(coors_org, self.secData[:,6], coors_mesh, method = 'nearest')
        V_mesh = hstack((Vx,Vy,Vz)) # interpolated velocity mat

        meshData = mat(zeros((pNum_x * pNum_z,6))) # initialize a mat storing interpolated meshData
        meshData[:,0:3] = coors_mesh
        meshData[:,3:6] = V_mesh
        self.meshData = meshData

    def meshITP_Nx(self,y_axis,z_axis): # interpolate the original wake data and project it onto a structral mesh
        """ y_axis should be a tuple indicating the min, max value and the number of segments, like y_axis = (-240, 240, 60), and similarly the z_axis """
        pNum_y = y_axis[2] + 1
        pNum_z = z_axis[2] + 1
        delta_y = (y_axis[1] - y_axis[0])/y_axis[2]
        delta_z = (z_axis[1] - z_axis[0])/z_axis[2]
        coor_y = range(int(y_axis[0]), int(y_axis[1] + delta_y), int(delta_y))
        coor_z = range(int(z_axis[0]), int(z_axis[1] + delta_z), int(delta_z))
        coor_x = sum(self.secData[:,1])/shape(self.secData[:,1])[0] # all y coordinate are the same, because this is a section of yNormal
        coors_org = self.secData[:,1:4] # build the original points mat

        # compute mesh coordinates and velocities
        from scipy.interpolate import griddata
        [Ymesh, Zmesh] = meshgrid(coor_y, coor_z)
        Xmesh = mat(ones((shape(Ymesh)))) * coor_x
        Xmesh = Xmesh.ravel()
        Ymesh = Ymesh.ravel()
        Zmesh = Zmesh.ravel()
        coors_mesh = (vstack((Xmesh,Ymesh,Zmesh))).T # build the mesh points mat

        Vx = griddata(coors_org, self.secData[:,4], coors_mesh, method = 'nearest')
        Vy = griddata(coors_org, self.secData[:,5], coors_mesh, method = 'nearest')
        Vz = griddata(coors_org, self.secData[:,6], coors_mesh, method = 'nearest')
        V_mesh = hstack((Vx,Vy,Vz)) # interpolated velocity mat

        meshData = mat(zeros((pNum_y * pNum_z,6))) # initialize a mat storing interpolated meshData
        meshData[:,0:3] = coors_mesh
        meshData[:,3:6] = V_mesh
        self.meshData = meshData


    def meshITP_Nz(self,x_axis,y_axis): # interpolate the original wake data and project it onto a structral mesh
        """ y_axis should be a tuple indicating the min, max value and the number of segments, like y_axis = (-240, 240, 60), and similarly the z_axis """
        pNum_x = x_axis[2] + 1
        pNum_y = y_axis[2] + 1
        delta_x = (x_axis[1] - x_axis[0])/x_axis[2]
        delta_y = (y_axis[1] - y_axis[0])/y_axis[2]
        coor_x = range(int(x_axis[0]), int(x_axis[1] + delta_x), int(delta_x))
        coor_y = range(int(y_axis[0]), int(y_axis[1] + delta_y), int(delta_y))
        coor_z = sum(self.secData[:,3])/shape(self.secData[:,3])[0] # all y coordinate are the same, because this is a section of yNormal
        coors_org = self.secData[:,1:4] # build the original points mat

        # compute mesh coordinates and velocities
        from scipy.interpolate import griddata
        [Xmesh, Ymesh] = meshgrid(coor_x, coor_y)
        Zmesh = mat(ones((shape(Ymesh)))) * coor_z
        Xmesh = Xmesh.ravel()
        Ymesh = Ymesh.ravel()
        Zmesh = Zmesh.ravel()
        coors_mesh = (vstack((Xmesh,Ymesh,Zmesh))).T # build the mesh points mat

        Vx = griddata(coors_org, self.secData[:,4], coors_mesh, method = 'nearest')
        Vy = griddata(coors_org, self.secData[:,5], coors_mesh, method = 'nearest')
        Vz = griddata(coors_org, self.secData[:,6], coors_mesh, method = 'nearest')
        V_mesh = hstack((Vx,Vy,Vz)) # interpolated velocity mat

        meshData = mat(zeros((pNum_x * pNum_y,6))) # initialize a mat storing interpolated meshData
        meshData[:,0:3] = coors_mesh
        meshData[:,3:6] = V_mesh
        self.meshData = meshData

    def x_cut(self, x):
        """ according to a x value, cut the meshData and extract the line data """
        cutData = {}
        index = where(self.meshData[:,0] == x)[0]
        cutData['y'] = self.meshData[index,1]
        cutData['z'] = self.meshData[index,2]
        cutData['Vx'] = self.meshData[index,-3]
        cutData['Vy'] = self.meshData[index,-2]
        cutData['Vz'] = self.meshData[index,-1]
        return cutData

    def y_cut(self, x):
        """ according to a x value, cut the meshData and extract the line data """
        cutData = {}
        index = where(self.meshData[:,1] == x)[0]
        cutData['x'] = self.meshData[index,0]
        cutData['z'] = self.meshData[index,2]
        cutData['Vx'] = self.meshData[index,-3]
        cutData['Vy'] = self.meshData[index,-2]
        cutData['Vz'] = self.meshData[index,-1]
        return cutData

    def z_cut(self, z):
        """ according to a z value, cut the meshData and extract the line data """
        cutData = {}
        index = where(self.meshData[:,2] == z)[0]
        cutData['x'] = self.meshData[index,0]
        cutData['y'] = self.meshData[index,1]
        cutData['Vx'] = self.meshData[index,-3]
        cutData['Vy'] = self.meshData[index,-2]
        cutData['Vz'] = self.meshData[index,-1]
        return cutData
