import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from numpy import arange

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.poly import create_vertice
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

class eddy():
    def __init__(self, dataset_path: str):    
        self.ds = xr.open_dataset(dataset_path)
        self.path = dataset_path
        self.data: List[Tuple] = self.load_data()
        self.hours_since_start_list: list = []
        self.grid_list: List[RegularGridDataset] = []
        self.days_list: list = [] 
        self.lon = [self.ds["LONGITUDE"].values.min(), self.ds["LONGITUDE"].values.max()]
        self.lat = [self.ds["LATITUDE"].values.min(), self.ds["LATITUDE"].values.max()]
        self.anticyclonic_list: list = []
        self.cyclonic_list: list = []
        self.masked_3d_array = None
        return None
    
    def start_axes(self, title):
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
        ax.set_xlim(self.lon[0], self.lon[1]), ax.set_ylim(self.lat[0], self.lat[1])
        ax.set_aspect("equal")
        ax.set_title(title, weight="bold")
        return ax


    def update_axes(self, ax, mappable=None):
        ax.grid()
        if mappable:
            plt.colorbar(mappable, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))

    def load_data(self):
        grid_collection = GridCollection.from_netcdf_cube(
            get_demo_path(self.path),
            "LONGITUDE",
            "LATITUDE",
            "TIME",
            heigth="ssh")
        return grid_collection.datasets
    
    def create_list_dataset(self, days_list: List):
        self.days_list = days_list
        for day in self.days_list:
            hours, g = self.data[day-1]
            g.vars['ssh'] = np.ma.array(g.vars['ssh'], mask=np.isnan(g.vars['ssh']))
            g.add_uv("ssh")
            g.bessel_high_filter("ssh", 500) 
            self.hours_since_start_list.append(hours)
            self.grid_list.append(g)

    def plot_graph(self, days:list):
        for day in days:
            ax = self.start_axes("SSH (m)")
            m = self.grid_list[day-1].display(ax, "ssh", vmin=-1, vmax=1, cmap="RdBu_r")
            self.update_axes(ax, m)
        return None
    def detect_eddies(self, min_pixel : int = 30):
        for day in self.days_list:
            date = datetime(1950, 1 , 1) + timedelta(hours=int(self.hours_since_start_list[day-1])) 
            a, c = self.grid_list[day-1].eddy_identification("ssh", "u", "v", date, 0.002, pixel_limit=(min_pixel, 2000), shape_error=70)
            self.anticyclonic_list.append(a)
            self.cyclonic_list.append(c)
        return None
    def plot_detected_eddies(self, days:list):
        for day in days:
            ax = self.start_axes("Detected Eddies")
            self.anticyclonic_list[day - 1].display(
                ax, color="r", linewidth=0.75, label="Anticyclonic ({nb_obs} eddies)", ref=-70
            )
            self.cyclonic_list[day - 1].display(ax, color="b", linewidth=0.75, label="Cyclonic ({nb_obs} eddies)", ref=-70)
            ax.legend()
            self.update_axes(ax)
        return None
    def generate_pixel_eddy(self, day):
        g = self.grid_list[day-1]
        a = self.anticyclonic_list[day-1]
        c = self.cyclonic_list[day-1]
        ssh = g.grid("ssh")
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
    
    
        