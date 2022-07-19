import numpy as np
import os
import xarray as xr
import yaml

parameters = yaml.safe_load(open('/home/ollie/ssunar/pyfiles/interpolator.yaml'))

output_path = parameters["output_path"]
filename = parameters["filename"]

def convert(input_dir, output_dir: str, output_file: str):
    input_file_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
        ])
    #input_file_paths.pop(29) #Uncomment these lines to exclude multiple months
    #input_file_paths.pop(11)
    data = xr.open_mfdataset(input_file_paths, combine = 'nested', concat_dim="TIME")
    data.to_netcdf(path=output_dir + "/" + output_file)
    return None

if __name__ == "__main__":
    convert(input_dir = output_path + "/months", output_dir = output_path, output_file = filename+".nc")