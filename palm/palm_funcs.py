import numpy as np
from netCDF4 import Dataset

""" basic data acquirement functions for PALM """

def getTime_palm_mask(dir, jobName, maskID, run_no_list):
    """ extract data of specific palm masked data """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tSeq_list.append(tSeq_tmp)

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(len(tSeq_list))], axis=0)
    tSeq = tSeq.astype(float)

    return tSeq



def getInfo_palm_mask(dir, jobName, maskID, run_no_list, var):
    """ extract data of specific palm masked data """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        
        tSeq_list.append(tSeq_tmp)

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(len(tSeq_list))], axis=0)
    tSeq = tSeq.astype(float)

    return tSeq, zSeq, ySeq, xSeq



def getData_palm_mask(dir, jobName, maskID, run_no_list, var, tInd, xInd, yInd, zInd):
    """ extract velocity data of specified probe groups """
    """ wait for opt """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []

    tInd_start = 0
    list_num = 0
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1

        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][zInd[0]:zInd[1]], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][yInd[0]:yInd[1]], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][xInd[0]:xInd[1]], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(list_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, zSeq, ySeq, xSeq, varSeq


def getWFData_palm(dir, jobName, run_no_list, var, tInd):

#    dir, jobName, run_no_list, var, tInd = jobDir, jobName, ['.000','.001','.002','.003'], 'rotor_power', [0,100]
    
    varList = ['turbine', 'time', 'x', 'y', 'z', 'rotor_diameter', 'tower_diameter', \
               'generator_power', 'generator_speed', 'generator_torque', 'pitch_angle', \
               'rotor_power', 'rotor_speed', 'rotor_thrust', 'rotor_torque', 'wind_direction', \
               'yaw_angle']
    
    dataDict = dict((i,[]) for i in varList)
    
    run_num = len(run_no_list)
    
    # read the output data of all run_no_list
    nc_file_list = []
    varSeq_list = []
    
    tInd_start = 0
    list_num = 0
    
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_wtm" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    
        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1
    
        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                dataDict['time'].append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                for var in ['generator_power', 'generator_speed', 'generator_torque', 'pitch_angle', 'rotor_power', \
                            'rotor_speed', 'rotor_thrust', 'rotor_torque', 'wind_direction', 'yaw_angle']:
                    dataDict[var].append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start], \
                                                  dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                dataDict['time'].append(tSeq_tmp[tInd[0]-tInd_start:])
                for var in ['generator_power', 'generator_speed', 'generator_torque', 'pitch_angle', 'rotor_power', \
                            'rotor_speed', 'rotor_thrust', 'rotor_torque', 'wind_direction', 'yaw_angle']:
                    dataDict[var].append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:], \
                                                  dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1
    
    # dimensions = list(nc_file_list[0].dimensions)
    # vars = list(nc_file_list[0].variables)
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable
    
    # wind turbines
    wtSeq = np.array(nc_file_list[0].variables['turbine']).astype(float)
    # z-coordinate of wind turbulines
    zSeq = np.array(nc_file_list[0].variables['z']).astype(float)
    # y-coordinate of wind turbulines
    ySeq = np.array(nc_file_list[0].variables['y']).astype(float)
    # z-coordinate of wind turbulines
    xSeq = np.array(nc_file_list[0].variables['x']).astype(float)
    # rotor diameter of wind turbines
    rdSeq = np.array(nc_file_list[0].variables['rotor_diameter']).astype(float)
    # tower diameter of wind turbines
    tdSeq = np.array(nc_file_list[0].variables['tower_diameter']).astype(float)
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([dataDict['time'][i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    for var in ['generator_power', 'generator_speed', 'generator_torque', 'pitch_angle', 'rotor_power', \
                            'rotor_speed', 'rotor_thrust', 'rotor_torque', 'wind_direction', 'yaw_angle']:
        dataDict[var] = np.concatenate([dataDict[var][i] for i in range(list_num)], axis=0)
        dataDict[var] = dataDict[var].astype(float)
    
    return tSeq, wtSeq, dataDict, zSeq, ySeq, xSeq, rdSeq, tdSeq


def palm_3d_single_time(dir, jobName, run_no, tInd, var):
    input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no + ".nc"
    input_data = Dataset(input_file, "r", format="NETCDF4")
    t = np.array(input_data.variables['time'][tInd], dtype=float)
    varSeq = np.array(input_data.variables[var][tInd], dtype=float)
    
    # extract the values of all dimensions of the var
    zName = list(input_data.variables[var].dimensions)[1] # the height name string
    zSeq = np.array(input_data.variables[zName][:], dtype=float) # array of height levels
    yName = list(input_data.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_data.variables[yName][:], dtype=float) # array of height levels
    xName = list(input_data.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_data.variables[xName][:], dtype=float) # array of height levels
    return t, zSeq, ySeq, xSeq, varSeq