import sys
import yaml

from generate_segmentation_mask import eddy

#Edit the yaml location for different parameters

yaml_file = sys.argv[2]

parameters = yaml.safe_load(open(yaml_file))

month = sys.argv[1]

ssh_data_addr = (parameters['output_path']+"/months/"+parameters["filename"]+"_"+str(parameters["year"])+'_'+str(1).zfill(3)+'_'+str(month).zfill(2)+'.nc')

eddy_instance = eddy(dataset_path=ssh_data_addr)
seg_data_addr = parameters["segmentation_masks_path"] + '/seg_mask_gridded_' + str(parameters["year"])+'_001_'+str(month).zfill(2)+'_new.nc'
eddy_instance.generate_mask(seg_data_addr)
