import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from numpy import arange

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.poly import create_vertice
from generic_eddy_class import eddy_detection
import re
from datetime import datetime, timedelta

from matplotlib.animation import FuncAnimation
from numpy import arange, isnan, meshgrid, ones

from py_eddy_tracker import start_logger
from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import GridCollection
from py_eddy_tracker.gui import GUI_AXES
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import os


from typing import List, Tuple

class eddy(eddy_detection):
    def __init__(self, dataset_path: str):    
        super().__init__(dataset_path)
        self.masked_3d_array = None
        return None
    
    def generate_pixel_eddy(self, day):
        g = self.grid_list[day-1]
        a = self.anticyclonic_list[day-1]
        c = self.cyclonic_list[day-1]
        ssh = g.grid("ssh")
        a.lon[a.lon>180] -= 360
        a.contour_lon_e[a.contour_lon_e>180] -= 360
        a.contour_lon_s[a.contour_lon_s>180] -= 360

        c.lon[c.lon>180] -= 360
        c.contour_lon_e[c.contour_lon_e>180] -= 360
        c.contour_lon_s[c.contour_lon_s>180] -= 360
        mask_a = np.zeros(ssh.shape, dtype="bool")
        x_a_name, y_a_name = a.intern(False)
        mask_c = np.zeros(ssh.shape, dtype="bool")
        x_c_name, y_c_name = c.intern(False)
        lon = g.x_c
        lon[lon<0] +=360
        g.x_c =lon
        for eddy in a:
            i, j = Path(create_vertice(eddy[x_a_name], eddy[y_a_name])).pixels_in(g)
            mask_a[i, j] = True

        for eddy in c:
            i, j = Path(create_vertice(eddy[x_c_name], eddy[y_c_name])).pixels_in(g)
            mask_c[i, j] = True
        return mask_a, mask_c
    def masking(self):
        masked_anticylonic = []
        masked_cyclonic = []
        masked_total = []
        for day in self.days_list:
            mask_a, mask_c = self.generate_pixel_eddy(day)
            masked_anticylonic.append(mask_a)
            masked_cyclonic.append(mask_c)
            masked_total.append(2 * mask_a + 1 * mask_c)
        masked_3d = np.dstack(masked_total)
        masked_3d = np.rollaxis(masked_3d,-1)
        return masked_3d
    def generate_mask(self, outfile: str):
        last_day = self.ds["TIME"].size
        days = list(range(1, last_day + 1))
        self.create_list_dataset(days)
        self.detect_eddies()
        self.masked_3d_array = self.masking()
        ds_seg_mask = self.ds.copy(deep = True)
        ds_seg_mask["seg_mask"] = ("TIME", "LONGITUDE", "LATITUDE"), self.masked_3d_array
        if os.path.isfile(outfile):
            os.remove(outfile)
        ds_seg_mask.to_netcdf(outfile)
        return None

if __name__ == "__main__":
    #Example of generating segmentation mask for a year
    for i in range(1,13):
        data_addr_nn = '/home/albedo/ssunar/ssh_filtered/months/ssh_gridded_1961_001_'+str(i).zfill(2)+'_new.nc'
        eddy_instance = eddy(dataset_path=data_addr_nn)
        outfile = "/home/albedo/ssunar/segmentation_masks/seg_mask_gridded_1961_001_"+str(i).zfill(2)+"_new.nc"
        eddy_instance.generate_mask(outfile)