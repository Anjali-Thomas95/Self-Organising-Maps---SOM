
"""

This code reads in three sets of data - weather@home data (both anthropogenic and natural forcing) and ERA5 data.
It preprocesses this data to prepare it for input into a Self-Organizing Map (SOM) algorithm. The datasets consist of Mean Sea Level Pressure (MSLP) data covering a geographic subset over New Zealand. The code flattens the data from three dimensions (latitude, longitude, time) to two dimensions (spatial, time) before feeding it into the SOM algorithm.

The output from the Self-Organizing Map (SOM) defines a set of patterns, each characterized by a specific number of rows and columns that user specify.
The SOM produces aset of representative weather patterns which are used later to classify ALL of the weather@home data
and these patterns are added to a netcdf file.

"""

# importing the needed python packages

import numpy as np   
import xarray as xr  # xarray is a package to interact with large geophysical datasets
import datetime
import matplotlib.pyplot as plt  # ploting package 
import os  # import os commands for making paths
import cartopy.crs as ccrs  # for plotting maps
import cartopy.feature as cfeat # for plotting maps
import seaborn as sns 
from glob import glob
import matplotlib.colors as colors

# this is the package that runs a Self ORganizinag Map on the data
from minisom import MiniSom    # specialised minisom calculation

# rather than creating a single function for reading, a seperate file that contains all the code for interacting with the weather@home data is created 
from read_weatherathome_data import read_weatherathome_sample
from sammon_algorithm import sammon

# these functions are used to write data into the xarray fromat and help in other places in the code
# this creates an xarray dataset with all the geographic patterns linked to specific nodes in a Self-Organizang Map
def create_xarray_som_patterns(som_number,latitude,longitude,global_latitude,global_longitude,variable_name1,variable1):
    d = {}
    d['som_number'] = ('som_number',som_number)
    d['latitude'] = ('latitude',latitude)
    d['longitude'] = ('longitude',longitude)
    d['global_latitude'] = (['latitude','longitude'], global_latitude)
    d['global_longitude'] = (['latitude','longitude'], global_longitude)

    d[variable_name1] = (['som_number','latitude','longitude'], variable1)
    dset = xr.Dataset(d) 
 
    return dset

# this creates an xarray dataset that fits with the format of the output from the weather@home dataset
def create_xarray(time,latitude,longitude,global_latitude,global_longitude,variable_name1,variable1):
    d = {}
    d['time'] = ('time',time)
    d['latitude'] = ('latitude',latitude)
    d['longitude'] = ('longitude',longitude)
    d['global_latitude'] = (['latitude','longitude'], global_latitude)
    d['global_longitude'] = (['latitude','longitude'], global_longitude)
    d[variable_name1] = (['time','latitude','longitude'], variable1)
    dset = xr.Dataset(d) 

    return dset

# this function takes in an x and y array and removes all the elements that are defined as Not A Number (NaN)
def no_nan_list(x,y):
    xnew=x[~isnan(y)]
    ynew=y[~isnan(y)]
    return xnew,ynew

# use Euclidean distance to calculate the BMU  (Best matching UNIT)
def calculate_bmu(test_array, input_array):    
    test_matrix=np.reshape(np.tile(test_array,(input_array.shape[1],input_array.shape[2])),(len(test_array),input_array.shape[1],input_array.shape[2]))
    return  np.sqrt(np.nansum(np.square(input_array-test_matrix),axis=0))

# create a histogram of the frequency of the various some nodes
def create_histogram(som_number):
    return np.histogram(som_number,bins=np.int(np.max(som_number))-np.int(np.min(som_number))+1)


################################################################################

#READING IN DATA

################################################################################

variable_name1='mslp'  # Mean sea level pressure
t00=datetime.datetime.now()

directoryname='/home/users/ath121/environmental/group_datasets/weatherathome/processed/batch117_ant/'
year=2013 # this is a constant

no_samples=41  # this defines how many realisations of the weather@home analysis are examined
t0=datetime.datetime.now()
ncid_ant=read_weatherathome_sample(directoryname,year,variable_name1,no_samples) # this reads in a randomised set of weather@home data to be used in the SOM 

latitude=ncid_ant['latitude'].values            
longitude=ncid_ant['longitude'].values            
global_latitude=ncid_ant['global_latitude'].values            
global_longitude=ncid_ant['global_longitude'].values            

t1=datetime.datetime.now()
print(t1-t0)

directoryname='/home/users/ath121/environmental/ajm226/weatherathome/processed/batch117_nat/' 

t0=datetime.datetime.now()
ncid_nat=read_weatherathome_sample(directoryname,year,variable_name1,no_samples) 

t1=datetime.datetime.now()
print(t1-t0)

# this section reads ERA5 reanalysis data which has been resampled to the w@h grid over New Zealand
directoryname='/home/users/ath121/environmental/group_datasets/ERA5/' 
t0=datetime.datetime.now()
files = glob(directoryname+'mslp_WAH_resampled_*.nc')
ncid_ERA5 = xr.open_mfdataset(files)

t1=datetime.datetime.now()
print(t1-t0)

ncid_ERA5_updated=create_xarray(ncid_ERA5.time,latitude,longitude,global_latitude,global_longitude,variable_name1,np.array(ncid_ERA5.mslp))

######################################################################################

## SOM CALCULATION STARTS HERE

######################################################################################
t11=datetime.datetime.now()
print(t11-t00)

# pre-rpocessing to get data in the right format!
# calculating the climatological mean geographic pattern using xarray
# this will allow us to remove the mean geographic pattern in each eventually

mslp_climatology_ant  = ncid_ant.mslp.mean('ensemble_number').mean('time')
mslp_climatology_ERA5 = ncid_ERA5_updated.mslp.mean('time')
mslp_climatology_nat  = ncid_nat.mslp.mean('ensemble_number').mean('time')

# Calculated the mean climatology from all 3 datasets

mslp_climatology = (mslp_climatology_ant+mslp_climatology_nat+mslp_climatology_ERA5)/3.0

# calculating the anomaly by taking the average from each dataset

ncid_ant_anom  = ncid_ant.mslp-mslp_climatology
ncid_nat_anom  = ncid_nat.mslp-mslp_climatology
ncid_ERA5_anom = ncid_ERA5_updated.mslp-mslp_climatology

#

# this is converting from a 3d matric to a 2d matrix (time,lat,lon) to (time,space)

input_mslp_ant  = np.reshape(ncid_ant_anom.values,(ncid_ant_anom.shape[0]*ncid_ant_anom.shape[1],ncid_ant_anom.shape[2]*ncid_ant_anom.shape[3]))
input_mslp_nat  = np.reshape(ncid_nat_anom.values,(ncid_nat_anom.shape[0]*ncid_nat_anom.shape[1],ncid_nat_anom.shape[2]*ncid_nat_anom.shape[3]))
input_mslp_ERA5 = np.reshape(ncid_ERA5_anom.values,(ncid_ERA5_anom.shape[0],ncid_ERA5_anom.shape[1]*ncid_ERA5_anom.shape[2]))


# concetatentae all of these datasets into one large matrix that will be input to the SOM scheme
input_mslp_field = np.concatenate((input_mslp_ant,input_mslp_nat,input_mslp_ERA5),axis=0)


################ The SOM algorithm

### define SOM size
som_col = 4
som_row = 3

print('Column = 4')
print('Row = 3')

som = MiniSom(x= som_col, y = som_row, input_len = input_mslp_field.shape[1], sigma=0.85, learning_rate=0.7)

# the learning rate and sigma goes between 0  and 1. These are defined in the paper by Korhonen!
#define random weights and train SOM
print("Training...")

t0=datetime.datetime.now()
som.train_random(input_mslp_field,input_mslp_field.shape[0])   # setting up random training scheme

qnt=som.quantization_error(input_mslp_field)  # statistical measure of how well the SOM represents the data
t1=datetime.datetime.now()
print(qnt,t1-t0)

[shape0,shape1]=input_mslp_field.shape

#identify winning SOM and the patterns 

windex = np.zeros((shape0,2))
for count,xx in enumerate(input_mslp_field):
    windex[count,:]=som.winner(xx)

som_number = (windex[:,0]*som_row)+windex[:,1]

# calculates the patterns for different regions

som_average = np.zeros((som_col*som_row,input_mslp_field.shape[1]))
for i in np.arange(0,som_col*som_row):
    som_average[i,:]=np.nanmean(input_mslp_field[som_number==i,:],axis=0)

som_pattern=np.reshape(som_average,(som_average.shape[0],ncid_ant.mslp.shape[2],ncid_ant.mslp.shape[3]))
print(som_average.shape)

################################################################
# PLOTTING TO MKAE SURE THE SOM WORKS
####################################################################

# the line below calcutes a histogram which tells you about the frequency of the different patterns

[values,bins]=create_histogram(som_number)

########### this codes plots the percentage frequency of each SOM pattern in a grid
dname = '/home/users/ath121/environmental/ath121/wah_som/som_sammon_trial/plots/'

plt.figure(figsize=(som_row,som_col))
gridded_values=np.reshape(values,(som_col,som_row))
weighted_grid=(gridded_values/len(som_number))*100.0
cmap=sns.cubehelix_palette(start=0.1,as_cmap=True)
g=sns.heatmap(weighted_grid, annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[])
plt.title(str(som_col)+'-'+str(som_row)+'Quantization error='+str(qnt))
plt.savefig(dname+'WAH_SOM5_percentage_'+str(som_col)+'_'+str(som_row)+'_'+str(no_samples)+'.png',dpi=900)     
plt.close()

########### this code plots the geographic patterns derived by the SOM

fig=plt.figure(figsize=(210/25.4,185/25.4))
k=0

for i in np.arange(0,som_col):
    for j in np.arange(0,som_row):
        ax1=plt.subplot(som_col,som_row,k+1,projection=ccrs.LambertCylindrical(central_longitude=180.0))
        ax1.add_feature(cfeat.LAND)
        cs=ax1.pcolormesh(global_longitude,global_latitude,som_pattern[k,:,:]/100.0,transform=ccrs.PlateCarree(),vmin=-20.0,vmax=+20.0,cmap='bwr')
#        ax1.contour(global_longitude,global_latitude, som_pattern[k,:,:], colors = "black")
        ax1.add_feature(cfeat.LAND)                
        ax1.set_extent([150, 200, -56, -20], ccrs.PlateCarree())
        ax1.coastlines()
        ax1.set_title('Node '+str(k+1), size=11)
        plt.axis('tight')
        print(k)
        k+=1

fig.subplots_adjust(right=0.75)
fig.suptitle('sigma=0.85, learning_rate=0.7', fontsize=13)
cs_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
norm = colors.BoundaryNorm(boundaries=np.linspace(-0.5,0.5,5), ncolors=5)
cbar=fig.colorbar(cs, cax=cs_ax,norm=norm)
cbar.set_label('Pressure (hPa)', rotation=90,fontsize=12)        
plt.savefig(dname+'WAH_SOM5_'+str(som_col)+'_'+str(som_row)+'_'+str(no_samples)+'.png',dpi=900)   

########### plotting the sammon map
rawdata = np.full([12, 65*70], 0) 
for i in range(12):
    rawdata[i,:] = np.array(som_pattern[i,:,:]).flatten()

s,e = sammon(rawdata, 2, inputdist = 'raw')

xx = s[:,0]
yy = s[:,1]
n = list(range(1,13))

fig, ax = plt.subplots(facecolor='w')
ax.scatter(xx, yy)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.axis('off')

for i, txt in enumerate(n):
    ax.annotate(txt, (xx[i], yy[i]))
    
plt.scatter(xx, yy, color='yellow')
fig.savefig(dname+'sammon_map_'+str(som_col)+'_'+str(som_row)+'_'+str(no_samples)+'.png', dpi=300)



#######################################################################
## WRITING THE DATA TO A NETCDF FILE USING XARRAY
########################################################################
# this final set of code writes xarray netcdf files but adds the mean back onto the anomaly data for each SOM cluster

som_number=np.arange(0,som_row*som_col)
variable_name1='mslp'    
directoryname='/home/users/ath121/environmental/ath121/wah_som/som_sammon_trial/'  # output directory
filename='WAH_SOM5_som_pattern_'+str(som_col)+'_'+str(som_row)+'_'+str(no_samples)+'.nc'  # output filename
# copies of mslp climatology multiplied so can add to anomaly data
mslp_climatology2=np.tile(mslp_climatology,som_pattern.shape[0])
mslp_climatology3=np.reshape(mslp_climatology2,(mslp_climatology2.shape[0],som_pattern.shape[0],mslp_climatology.shape[1]))
mslp_climatology4=np.moveaxis(mslp_climatology3,1,0)
som_xarray=create_xarray_som_patterns(som_number,latitude,longitude,global_latitude,global_longitude,variable_name1,som_pattern+mslp_climatology4)
som_xarray.to_netcdf(directoryname+filename)

# this final set of code writes xarray netcdf files but jsut writes the anomaly data associated with each SOM cluster
variable_name1='mslp'    
filename='WAH_SOM5_som_anom_pattern_'+str(som_col)+'_'+str(som_row)+'_'+str(no_samples)+'.nc'
som_anom_xarray=create_xarray_som_patterns(som_number,latitude,longitude,global_latitude,global_longitude,variable_name1,som_pattern)
som_anom_xarray.to_netcdf(directoryname+filename)

#################################################################################################################################################################################



