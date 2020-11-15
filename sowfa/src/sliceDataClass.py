import numpy as np
from statistics import mode
from scipy.interpolate import griddata

class Slice:
    ''' 主要用于对某个平面信息进行网格化插值 '''
    """ members """
    # sliceData = 0
    # meshData = 0
    # meshData_horizAve = 0

    ''' constructor '''
    def __init__(self, sliceData, axis_): # constructor
        self.data = sliceData # sliceData is the extracted data by sliceDataExt.py
        self.axis = axis_ # x:0, y:1, z:2
        self.wash()
        self.tSeq = self.data['time']
        self.tNum = self.tSeq.size
        self.pNum = self.data['pNo'].size

    ''' functions '''
    def get_wakeData(self, sliceData): # function for reassigning
        self.data = sliceData

    def wash(self):
        """ somehow the original sliceData has some dirty points (e.g. for slice y = 1000 some points' y-coordinate is not 1000), we have to wash them out """
        data = self.data['point'][:,self.axis]
        rightValue = mode(data)
        self.N_location = rightValue
        delInd = np.where(data != rightValue)[0]

        self.data['point'] = np.delete(self.data['point'], delInd, axis=0)

        pNum = self.data['point'].shape[0]
        pNoArray = np.array([[i] for i in range(pNum)])
        self.data['pNo'] = pNoArray

        tNum = self.data['time'].size

        for scalar in self.data['scalars']:
            tmp = []
            for i in range(tNum):
                tmp.append(np.delete(self.data[scalar][i], delInd, axis=0))
            self.data[scalar] = np.array(tmp)

        for vector in self.data['vectors']:
            tmp = []
            for i in range(tNum):
                tmp.append(np.delete(self.data[vector][i], delInd, axis=0))
            self.data[vector] = np.array(tmp)


    def p_nearest(self, p_coor):
        coor_org = np.copy(self.data['point'][:])
        coor_org[:,0] = coor_org[:,0] - p_coor[0]
        coor_org[:,1] = coor_org[:,1] - p_coor[1]
        coor_org[:,2] = coor_org[:,2] - p_coor[2]
        d2 = [sum(np.power(coor_org[i,:],2)) for i in range(self.pNum)]
        d2 = np.array(d2)
        d2_min = np.min(d2)
        pNearest = np.where(d2 == d2_min)[0]
        return int(pNearest), np.sqrt(d2_min)


    def pITP_Nz(self, p_coor, varName, method_='nearest'):
        """ interpolate to get the time series of a certain point """
        coor_org = np.copy(self.data['point'][:,0:2])
        vSeq = []
        for tInd in range(self.tNum):
            print(tInd)
            var = self.data[varName][tInd,:]
            v = griddata(coor_org, var, p_coor, method=method_)
            vSeq.append(v)
        vSeq = np.array(vSeq)
        return vSeq


    def meshITP_Ny(self,x_axis,z_axis,varArray,method_='cubic'): # interpolate the original wake data and project it onto a structral mesh
        """ x_axis should be a tuple indicating the min, max value and the number of segments, like x_axis = (-240, 240, 60), and similarly the z_axis """
        pNum_x = x_axis[2] + 1
        pNum_z = z_axis[2] + 1
        delta_x = (x_axis[1] - x_axis[0])/x_axis[2]
        delta_z = (z_axis[1] - z_axis[0])/z_axis[2]
        coor_x = range(int(x_axis[0]), int(x_axis[1] + delta_x), int(delta_x))
        coor_z = range(int(z_axis[0]), int(z_axis[1] + delta_z), int(delta_z))
        coors_org = self.data['point'][:,0:3:2]

        # compute mesh coordinates and corresponding scalar or vector values
        [Xmesh, Zmesh] = np.meshgrid(coor_x, coor_z)
        Xmesh = Xmesh.ravel()
        Zmesh = Zmesh.ravel()
        coors_mesh = (np.vstack((Xmesh,Zmesh))).T # build the mesh points mat

        varArray_mesh = griddata(coors_org, varArray, coors_mesh, method=method_)

        xticks = np.array(coor_x)
        zticks = np.array(coor_z)

        var = []
        for zind in range(pNum_z):
            var.append(varArray_mesh[zind*pNum_x:(zind+1)*pNum_x])
        var = np.array(var)

        return xticks, zticks, var


    def meshITP_Nx(self,y_axis,z_axis,varArray,method_='cubic'): # interpolate the original wake data and project it onto a structral mesh
        pNum_y = y_axis[2] + 1
        pNum_z = z_axis[2] + 1
        delta_y = (y_axis[1] - y_axis[0])/y_axis[2]
        delta_z = (z_axis[1] - z_axis[0])/z_axis[2]
        coor_y = range(int(y_axis[0]), int(y_axis[1] + delta_y), int(delta_y))
        coor_z = range(int(z_axis[0]), int(z_axis[1] + delta_z), int(delta_z))
        coors_org = self.data['point'][:,1:3]

        # compute mesh coordinates and corresponding scalar or vector values
        [Ymesh, Zmesh] = np.meshgrid(coor_y, coor_z)
        Ymesh = Ymesh.ravel()
        Zmesh = Zmesh.ravel()
        coors_mesh = (np.vstack((Ymesh,Zmesh))).T # build the mesh points mat

        varArray_mesh = griddata(coors_org, varArray, coors_mesh, method=method_)

        yticks = np.array(coor_y)
        zticks = np.array(coor_z)

        var = []
        for zind in range(pNum_z):
            var.append(varArray_mesh[zind*pNum_y:(zind+1)*pNum_y])
        var = np.array(var)

        return yticks, zticks, var


    def meshITP_Nz(self,x_axis,y_axis,varArray,method_='cubic'): # interpolate the original wake data and project it onto a structral mesh
        pNum_x = x_axis[2] + 1
        pNum_y = y_axis[2] + 1
        delta_x = (x_axis[1] - x_axis[0])/x_axis[2]
        delta_y = (y_axis[1] - y_axis[0])/y_axis[2]
        coor_x = range(int(x_axis[0]), int(x_axis[1] + delta_x), int(delta_x))
        coor_y = range(int(y_axis[0]), int(y_axis[1] + delta_y), int(delta_y))
        coors_org = self.data['point'][:,0:2]

        # compute mesh coordinates and corresponding scalar or vector values
        [Xmesh, Ymesh] = np.meshgrid(coor_x, coor_y)
        Xmesh = Xmesh.ravel()
        Ymesh = Ymesh.ravel()
        coors_mesh = (np.vstack((Xmesh,Ymesh))).T # build the mesh points mat

        varArray_mesh = griddata(coors_org, varArray, coors_mesh, method=method_)

        xticks = np.array(coor_x)
        yticks = np.array(coor_y)

        var = []
        for yind in range(pNum_y):
            var.append(varArray_mesh[yind*pNum_x:(yind+1)*pNum_x])
        var = np.array(var)

        return xticks, yticks, var
