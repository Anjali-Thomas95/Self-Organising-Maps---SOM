
"""
This file has a set of functions used in the SOM algorithm code

"""
# import packages
import numpy as np
import pandas as pd
import xarray as xr
import os  
from glob import glob
import random

# use euclidean distance to calculate the BMU
def calculate_bmu(test_array, input_array):    
    test_matrix=np.reshape(np.tile(test_array,(input_array.shape[1],input_array.shape[2])),(len(test_array),input_array.shape[1],input_array.shape[2]))
    return  np.sqrt(np.nansum(np.square(input_array-test_matrix),axis=0))


# use euclidean distance to calculate the BMU 4D data
def calculate_bmu4d(test_array, input_array):    
    test_component1=np.tile(test_array,(input_array.shape[0]))
    test_component2=np.reshape(test_component1,(test_array.shape[0],test_array.shape[1],input_array.shape[0],test_array.shape[2]))
    input_component1=np.tile(input_array,(test_array.shape[0]))
    input_component2=np.reshape(input_component1,(input_array.shape[0],input_array.shape[1],test_array.shape[0],input_array.shape[2]))
    input_component3=np.moveaxis(input_component2,[0,1,2,3],[2,1,0,3])
    return  np.sqrt(np.nansum(np.nansum(np.square(input_component3-test_component2),axis=1),axis=2))

#Create a dataset. The dictionnary keys are the variables contained in the Dataset.
def create_xarray_all(time,ensemble_number,latitude,longitude,global_latitude,global_longitude,variable_name1,variable1):
    d = {}
    d['time'] = ('time',time)
    d['ensemble_number'] = ('ensemble_number',ensemble_number)
    d['latitude'] = ('latitude',latitude)
    d['longitude'] = ('longitude',longitude)
    d['global_latitude'] = (['latitude','longitude'], global_latitude)
    d['global_longitude'] = (['latitude','longitude'], global_longitude)

    d[variable_name1] = (['time','ensemble_number','latitude','longitude'], variable1)
    dset = xr.Dataset(d) 

    return dset

#Create a dataset. The dictionnary keys are the variables contained in the Dataset.
def create_xarray(time,ensemble_number,variable_name1,variable1):
    d = {}
    d['time'] = ('time',time)
    d['ensemble_number'] = ('ensemble_number',ensemble_number)  
    d[variable_name1] = (['time','ensemble_number'], variable1)
    dset = xr.Dataset(d) 

    return dset

#Create a dataset. The dictionnary keys are the variables contained in the Dataset.
def create_time_xarray(time,variable_name1,variable1):
    d = {}
    d['time'] = ('time',time)
    d[variable_name1] = (['time'], variable1)

    dset = xr.Dataset(d) 
    return dset

#Create a dataset. The dictionnary keys are the variables contained in the Dataset.
def create_xarray_histogram(time,ensemble_number,bins,variable_name1,variable1):
    d = {}

    d['time'] = ('time',time)
    d['ensemble_number'] = ('ensemble_number',ensemble_number)
    d['bins'] = ('bins',bins)
   
    d[variable_name1] = (['time','ensemble_number','bins'], variable1)
    dset = xr.Dataset(d) 

    return dset

# TEST Read w@h data and give the regional mean
def read_weatherathome_regional_mean_test(directoryname,year,variable_name1='tmax'):

    files = glob(directoryname+'*'+str(year)+'*.nc')

    list_of_arrays=[]
    ensemble_member=[]

    for i,file in enumerate(files[0:100]):
        ds = xr.open_dataset(file)
        variable=ds[variable_name1].mean('latitude0').mean('longitude0')

        if(variable.shape[0]==360):
            list_of_arrays.append(variable.values)
            ensemble_member.append(i)

    vals = np.concatenate(list_of_arrays, axis=1)
    time=ds['time0'].values            

    return create_xarray(time,ensemble_member,variable_name1,vals)


# Read w@h data and give the regional mean
def read_weatherathome_regional_mean(directoryname,year,variable_name1='tmax'):

    files = glob(directoryname+'*'+str(year)+'*.nc')
    if(variable_name1=='wind_speed'):
        list_of_arrays=[]
        ensemble_member=[]

        for i,file in enumerate(files):
            ds = xr.open_dataset(file)
            variable=ds[variable_name1].mean('latitude1').mean('longitude1')

            if(variable.shape[0]==360):
                list_of_arrays.append(variable.values)
                ensemble_member.append(i)
        vals = np.concatenate(list_of_arrays, axis=1)

        time=ds['time0'].values            

    else:

        list_of_arrays=[]
        ensemble_member=[]

        for i,file in enumerate(files[:40]):
            ds = xr.open_dataset(file)
            variable=ds[variable_name1].mean('latitude0').mean('longitude0')

            if(variable.shape[0]==360):
                list_of_arrays.append(variable.values)
                ensemble_member.append(i)

        vals = np.concatenate(list_of_arrays, axis=1)
        time=ds['time0'].values            

    return create_xarray(time,ensemble_member,variable_name1,vals)


# TEST Read w@h data and give the histogram
def read_weatherathome_pdf_test(directoryname,year,variable_name1='tmax'):

    files = glob(directoryname+'*'+str(year)+'*.nc')
    start=0.0
    stop=0.002

    stepsize=stop/400

    list_of_arrays=[]
    ensemble_member=[]

    for i,file in enumerate(files[0:10]):
        ds = xr.open_dataset(file)
        variable=ds[variable_name1]

        if(variable.shape[0]==360):
            ensemble_member.append(i)
            for dayno in np.arange(0,360):
                tmp=(variable.values)
                tmp_flat=np.reshape(tmp[dayno,0,:,:],(tmp.shape[2]*tmp.shape[3],1))
                [height,bin]=np.histogram(tmp_flat,np.arange(start,stop,stepsize))
                list_of_arrays.append(height)

    vals = np.concatenate(list_of_arrays, axis=0)
    ensemble_member=np.array(ensemble_member)
    variable=np.reshape(vals,(np.int(vals.shape[0]/(399*ensemble_member.shape[0])),ensemble_member.shape[0],399))

    time=ds['time0'].values            

    return    create_xarray_histogram(time,ensemble_member,np.arange(start,stop-stepsize,stepsize),variable_name1,variable)

# Read w@h data 
def read_weatherathome_sample(directoryname,year,variable_name1='tmax',no_samples=41):

    files = glob(directoryname+'*'+str(year)+'*.nc')

    sample=random.sample(range(0,len(files)),no_samples)

    print(sample) 

    list_of_arrays=[]
    ensemble_member=[]    

    for i,file_number in enumerate(sample):
        ds = xr.open_dataset(files[file_number])

        variable=ds[variable_name1]

        if(variable.shape[0]==360):
            ensemble_member.append(i)
            list_of_arrays.append(variable.values)
    
    vals = np.concatenate(list_of_arrays, axis=0)
    ensemble_member=np.array(ensemble_member)

    time=ds['time0'].values            
    latitude=ds['latitude0'].values       
    longitude=ds['longitude0'].values            

    variable=np.reshape(vals,(time.shape[0],ensemble_member.shape[0],latitude.shape[0],longitude.shape[0]))
    
    global_latitude=ds['global_latitude0'].values            
    global_longitude=ds['global_longitude0'].values            

    return    create_xarray_all(time,ensemble_member,latitude,longitude,global_latitude,global_longitude,variable_name1,variable)

# Read w@h data and match the BMU
def read_weatherathome_bmu_match(directoryname,year,som_xarray,variable_name1='mslp'):    

    files = glob(directoryname+'*'+str(year)+'*.nc')
    bmu_output=[]
    test_array=som_xarray.mslp.values
    ensemble_member=[]    

    for i,file in enumerate(files[:40]):
        ds = xr.open_dataset(file)
        variable=ds[variable_name1]

        if(variable.shape[0]==360):
            ensemble_member.append(i)
            input_array=np.squeeze(variable.values)        
            bmu1=calculate_bmu4d(test_array, input_array)
            time=ds['time0'].values
            bmu_output.append(bmu1)

    print(np.array(bmu_output).shape)
    bmu_min=np.argmin(bmu_output,axis=1)
    variable_name1='bmu'

    bmu_xarray=create_xarray(time,ensemble_member,variable_name1,bmu_min.T)

    return np.array(bmu_output),bmu_xarray

# Read w@h data
def read_weatherathome(directoryname,year,variable_name1='tmax'):

    files = glob(directoryname+'*'+str(year)+'*.nc')

    list_of_arrays=[]
    ensemble_member=[]

    for i,file in enumerate(files[:3000]):
        ds = xr.open_dataset(file)
        variable=ds[variable_name1]

        if(variable.shape[0]==360):
            ensemble_member.append(i)
            list_of_arrays.append(variable.values)

    vals = np.concatenate(list_of_arrays, axis=1)
    ensemble_member=np.array(ensemble_member)

    time=ds['time0'].values
    latitude=ds['latitude0'].values
    longitude=ds['longitude0'].values

    variable=np.reshape(vals,(time.shape[0],ensemble_member.shape[0],latitude.shape[0],longitude.shape[0]))

    global_latitude=ds['global_latitude0'].values
    global_longitude=ds['global_longitude0'].values

    return    create_xarray_all(time,ensemble_member,latitude,longitude,global_latitude,global_longitude,variable_name1,variable)

