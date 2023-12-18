#!/usr/bin/env anaconda2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:03:28 2022

@author: Bivek Panthi, Shishir Sunar
"""
import os
import sys
import yaml

from interpolator_func import*
import month_to_year as m2y

#Edit the yaml location for different parameters

yaml_file = sys.argv[2]

parameters = yaml.safe_load(open(yaml_file))

month = sys.argv[1]

nc_or_nod2d = int(parameters["nc_or_nod2d"])
input_path_data_nc = None
input_path_grid_nc = None

input_path_nod2dfile = None
input_path_elm2dfile = None

input_left = parameters["input_left"]
input_right = parameters["input_right"]
input_top = parameters["input_top"]
input_bottom = parameters["input_bottom"]
year = parameters["year"]
filename = parameters["filename"]

output_path_months = parameters["output_path"] 
if not os.path.isdir(output_path_months):
    os.makedirs(output_path_months, exist_ok=True)
    # os.mkdir(output_path_months)



#checking if the path address is valid or not
if nc_or_nod2d == 1:
    input_path_data_nc = parameters["input_path_data_nc"]
    input_path_grid_nc = parameters["input_path_grid_nc"]
    if not os.path.exists(input_path_data_nc):
        print(input_path_data, "doesn't exist!")
        sys.exit(1)
    if not os.path.exists(input_path_grid_nc):
        print(input_path_grid, "doesn't exist!")
        sys.exit(1)
else:
    input_path_nod2dfile = parameters["input_path_nod2dfile"]
    input_path_elm2dfile = parameters["input_path_elm2dfile"]
    if not os.path.exists(input_path_nod2dfile):
        print(input_path_nod2dfile, "doesn't exist!")
        sys.exit(1)
    if not os.path.exists(input_path_elm2dfile):
        print(input_path_elm2dfile, "doesn't exist!")
        sys.exit(1)

interpolator_object = Interpolator(input_path_data_nc, input_path_grid_nc, year, month, input_left, input_right, input_top, input_bottom, input_path_nod2dfile, input_path_elm2dfile, nc_or_nod2d)
interpolator_object.nn_interpolation_action(output_path_months, filename, mask_flag = 1)


