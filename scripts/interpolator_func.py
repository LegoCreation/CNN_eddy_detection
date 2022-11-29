#!/usr/bin/env anaconda2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:14:36 2022

@author: Bivek panthi, Shishir Sunar
"""

"""
Important:
1. Please pass the correct data_address, grid_address, nod2dfile in yaml file.
3. Please change the resolution value in region_init function.
5. If you want to convert NN interpolated data to mask array with mask from linear interpolated data, set 
    mask_flag to 1 in nn_interpolation_action function.
"""

import xarray as xr
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
import scipy.spatial.qhull as qhull
import matplotlib.tri as mtri
import matplotlib.cm as cm
import os
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
from pandas import Timestamp, to_datetime
from filters import bessel_high_filter

#import dask.dataframe as dd
xr.set_options(display_style="text")

class Interpolator:
    def __init__(self, data_address:str , grid_address: str, year: int, month: int, left: int, right: int, top: int, bottom: int, nod2dfile: str = None, elm2dfile: str = None, nc_flag: int = 1, bessel_filter: int = 0):
        """
        Initilizer function
        arguments:
        -----------
        data_address: the absolute address of the data
        grid_address: the absolute address of the grid
        year: the year for which you want to run the interpolation
        month: to reduce the task each year is sub-divided into months. It will be automatically passed in slurm script
        left, right, top, bottom: values in degree
        """

        self.nc_flag = nc_flag
        self.bessel_filter = bessel_filter
        self.data = xr.open_dataset(data_address)
        if self.nc_flag:
            self.grid = xr.open_dataset(grid_address)
            self.model_lon = self.grid.lon.values
            self.model_lat = self.grid.lat.values
        else:
            file_content_nod2d = pd.read_csv(
                nod2dfile,
                delim_whitespace=True,
                skiprows=1,
                names=["node_number", "x", "y", "flag"],
            )
            
            file_content_elem = pd.read_csv(
                elm2dfile,
                delim_whitespace=True,
                skiprows=1,
                names=["first_elem", "second_elem", "third_elem"],
            )
            
            self.model_lon = file_content_nod2d.x.values
            self.model_lat = file_content_nod2d.y.values
            self.elem = file_content_elem.values - 1

        self.lon, self.lat = self.region_init(left, right, top, bottom)
        lon2, lat2 = np.meshgrid(self.lon, self.lat)
        self.lon2 = lon2.T
        self.lat2 = lat2.T
        self.year = year
        self.month = month

        

    """
    def __init__(self, data_address, grid_address, year, month, left, right, top, bottom):
        
        Initilizer function
        arguments:
        -----------
        data_address: the absolute address of the data
        grid_address: the absolute address of the grid
        year: the year for which you want to run the interpolation
        month: to reduce the task each year is sub-divided into months. It will be automatically passed in slurm script
        left, right, top, bottom: values in degree
        
        
        self.data = xr.open_dataset(data_address) 
        self.grid = xr.open_dataset(grid_address) 
        self.model_lon = self.grid.lon.values
        self.model_lat = self.grid.lat.values
        self.lon, self.lat = self.region_init(left, right, top, bottom)
        lon2, lat2 = np.meshgrid(self.lon, self.lat)
        self.lon2 = lon2.T
        self.lat2 = lat2.T
        self.year = year
        self.month = month
    """

    def region_init(self, left, right, top, bottom, res: float = 1/12):
        """
        This function specifies the region. (longitude and latitude sequences)
        arguments:
        -----------
        left, right, top, bottom: values in degree
        
        return:
        -----------
        longitude and latitude of the region
        """
        
        """
        if you want to input the 'left', 'right', 'bottom' and 'top' parameters then uncomment the code below
        flag_input = 1
        #defining our region
        print("Please input values in degree")
        print("Please enter with '-' sign if the values are in degree south or west")
        while(flag_input):
            left = input("left: ")
            right = input("right: ")
            top = input("top: ")
            bottom = input("bottom: ")
        """
        try: 
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
        except ValueError:
            raise TypeError("Only integers are allowed")

        lon = np.arange(left, right, res)
        lat = np.arange(top, bottom, res)
        return lon, lat
    
    def data_triangulation_init(self):
        """
        return:
        -----------
        tri: the default matplotlib.tri.TriFinder of this triangulation
        triang: triangulation
        
        Please note that you cannot pickle the tri object (as it is a C++ object). So, you cannot apply dask.delayed to
        this function.
        """
        if self.nc_flag:
            elements = (self.grid.elements.data.astype('int32') - 1).T
        else:
            elements = self.elem
        d = self.model_lon[elements].max(axis=1) - self.model_lon[elements].min(axis=1)
        no_cyclic_elem = np.argwhere(d < 100).ravel()
        triang = mtri.Triangulation(self.model_lon, self.model_lat, elements[no_cyclic_elem])
        tri = triang.get_trifinder()
        return tri, triang
    
    def data_triangulation(self, tri, triang, data_sample):
        triangularized_data = mtri.LinearTriInterpolator(triang, data_sample,trifinder=tri)
        return triangularized_data
    
    def nearest_nei_interpolator(self, data_sample, points):  
        #points = np.vstack((self.model_lon, self.model_lat)).T
        nn_interpolation = NearestNDInterpolator(points, data_sample)
        interpolated_nn_fesom = nn_interpolation((self.lon2, self.lat2))
        return interpolated_nn_fesom
               
    def nearest_nei_interpolator_mask(self, masked_data, data_sample, points):   
        #points = np.vstack((self.model_lon, self.model_lat)).T
        nn_interpolation = NearestNDInterpolator(points, data_sample)
        interpolated_nn_fesom = nn_interpolation((self.lon2, self.lat2))
        mask = masked_data.mask
        interpolated_nn_fesom_masked = np.ma.array(interpolated_nn_fesom, mask=mask)
        return interpolated_nn_fesom_masked
    
    def graph(self, graph_data):
        plt.figure(figsize=(10,10))
        plt.imshow(np.flipud(graph_data), cmap=cm.seismic, vmin=-1.5, vmax=0.5)
        plt.colorbar(orientation='horizontal', pad=0.04)
    
        
        
    def linear_interpolation_action(self, out_path, filename):
        """
        Creates linear interpolated object of each month and calls to_netcdf function to save the object in netcdf4 file
        """
        linear_interpolator_list = []
        tri_triang = self.data_triangulation_init()
        days_dict = {0:0, 1:31, 2:59 ,3:90, 4:120, 5:151, 6:181, 7:212, 8:243, 9:273, 10:304, 11:334, 12:365}
        if not(int(self.year) % 4):
            days_dict = {0:0, 1:31, 2:60 ,3:91, 4:121, 5:152, 6:182, 7:213, 8:244, 9:274, 10:305, 11:335, 12:366}
        prev_month_days = days_dict[int(self.month)-1]
        days = days_dict[int(self.month)] - prev_month_days
        
        for day in range(days):
            data_sample = self.data.ssh[(day+prev_month_days),:].values
            triangularized_data = self.data_triangulation(tri_triang[0], tri_triang[1], data_sample)
            linear_interpolator_list.append(np.ma.masked_invalid(triangularized_data(self.lon2, self.lat2)))
        print("1122")
        self.to_netcdf(linear_interpolator_list, out_path, filename, days)
        
        
    def nn_interpolation_action(self, out_path, filename, mask_flag: int = 0):
        """
        Creates NN interpolated object of one year and calls to_netcdf function to save the object in netcdf4 file
        parameter:
        -----------------
        mask_flag: if set to 1, converts NN interpolated data to mask array with mask from linear interpolated data
        by default set to 0
        """
        nn_interpolator_list = []
        #days_dict = {1:31, 2:28 ,3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        
        days_dict = {0:0, 1:31, 2:59 ,3:90, 4:120, 5:151, 6:181, 7:212, 8:243, 9:273, 10:304, 11:334, 12:365}
        if not(int(self.year) % 4): #if leap year
            days_dict = {0:0, 1:31, 2:60 ,3:91, 4:121, 5:152, 6:182, 7:213, 8:244, 9:274, 10:305, 11:335, 12:366}
        prev_month_days = days_dict[int(self.month)-1]
        days = days_dict[int(self.month)] - prev_month_days       
        
        points = np.vstack((self.model_lon, self.model_lat)).T
        
        if mask_flag == 0:
            for day in range(days):
                data_sample = self.data.ssh[(day+prev_month_days),:].values
                nn_interpolator = dask.delayed(self.nearest_nei_interpolator)(data_sample, points)
                nn_interpolator_list.append(nn_interpolator)
                
        elif mask_flag == 1:
            tri_triang = self.data_triangulation_init()
            data_sample = self.data.ssh[0,:].values
            triangularized_data = self.data_triangulation(tri_triang[0], tri_triang[1], data_sample)
            #We can reuse same triangularized_data as mask will be the same for all timesteps
            for day in range(days):
                data_sample = self.data.ssh[(day+prev_month_days),:].values
                nn_interpolator = dask.delayed(self.nearest_nei_interpolator_mask)(triangularized_data(self.lon2, self.lat2), data_sample, points)
                nn_interpolator_list.append(nn_interpolator)
 
        nn_interpolator_list = dask.compute(*nn_interpolator_list)
        nn_interpolation = []
        for i in range(days):
            nn_interpolation.append(np.ma.masked_invalid(nn_interpolator_list[i]))
        
        self.to_netcdf(nn_interpolation,out_path, filename, days)
    
    def to_netcdf(self, interpolator_data, out_path, filename, days: int = 30) -> None:
        """
        This function saves the interpolated data to a netcdf4 file. Please change the out_path to your preffered location. 
        
        parameters:
        -----------
        linear_interpolator: The final linear interpolated object
        days: default set to 30
        """
        outfile = (out_path+"/"+filename+"_"+str(self.year)+'_'+str(0+1).zfill(3)+'_'+str(self.month).zfill(2)+'.nc')
        if os.path.isfile(outfile):
            os.remove(outfile)
            
        times = self.data.coords['time'].values
        time_unit_out= "hours since " + str(self.year) + "-01-01 00:00:00" #hours in case higher resolution output
        
        fw = Dataset(outfile, 'w')
        
        #Dimensions
        fw.createDimension('TIME', days)
        fw.createDimension('LONGITUDE', self.lon.size)
        fw.createDimension('LATITUDE', self.lat.size)
        
        #Variables
        lon3 = fw.createVariable('LONGITUDE', 'f', ('LONGITUDE',))
        lat3  = fw.createVariable('LATITUDE', 'f', ('LATITUDE',))        
        time = fw.createVariable('TIME', 'f', ('TIME',))                  
        ssh = fw.createVariable('ssh',np.float64, ('TIME','LONGITUDE', 'LATITUDE'))
        
        
        #Setting units and descriptions
        lat3.units = 'degrees_north'
        lat3.long_name = 'latitude'
        lon3.units = 'degrees_east'
        lon3.long_name = 'longitude'
        ssh.units = 'm'
        ssh.description = 'sea surface elevation'
        ssh.long_name = 'sea surface elevation'
        time.setncattr('unit',time_unit_out)
        
        #Filtered ssh using bessel filter
        if self.bessel_filter:
            ssh_bessel = fw.createVariable('ssh_bessel',np.float64, ('TIME','LONGITUDE', 'LATITUDE'))
            ssh_bessel.units = 'm'
            ssh_bessel.description = 'sea surface elevation bessel filtered'
            ssh_bessel.long_name = 'sea surface elevation bessel'
        
        #Storing co-ordinates
        lon3[:] = self.lon[:]
        lat3[:] = self.lat[:]
        
        #Masked co-ordinates for the filtering process later
        lat_ma = np.ma.array(lat3[:])
        lon_ma = np.ma.array(lon3[:])
        for day in range(days): 
            ts=Timestamp(times[day])
            t=ts.to_pydatetime()
            ssh[day] = interpolator_data[day][:]
            if self.bessel_filter:
                ssh_bessel[day] = bessel_high_filter(interpolator_data[day][:], wave_length = 500, y_c = lat_ma, x_c = lon_ma)
            time[day] = date2num(t, time_unit_out)
        
        fw.close()
        return None 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
