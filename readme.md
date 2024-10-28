# Detection of Mesoscale eddies using Convolutional Neural Network on Simulated Data 

## File Structure
```
\--CNN_eddy_detection
        \--static # All python scripts 
        \--notebooks # All jupyter notebooks    
        \--test #test notebooks or scripts
        -- interpolator.sh                    # interpolator slurm script
        -- README.md
        -- requirements.txt          # Required dependencies to run this program
        -- .gitignore    
```

## Requirement Installation guide

```bash
# Clone the repo.
git clone https://github.com/LegoCreation/CNN_eddy_detection
cd CNN_eddy_detection

# Create environment to work in and activate it
$ conda create -n eddy-tracking python=3.8.16
$ conda activate eddy-tracking


# use pip to install PyEddyTracker
$ pip install pyEddyTracker==3.6.1

# manually install/uninstall a couple of dependencies
$ pip uninstall pymc3
$ pip uninstall pymc-learn
$ pip install -r pip_requirements.txt
$ pip install xarray==2022.11.0
$ pip install numpy==1.21.0
$ pip install zarr==2.13.3
$ pip install dask==2023.2.0

## For tensorflow numpy==1.24.3 is required but for py-eddy-tracker numpy==1.21.0 is required. Hence cannot be used simultaneouly.


# Create a Kernel for jupyter notebook
$ mamba install ipykernel
$ python -m ipykernel install --user --name eddy-tracking --display-name="eddy-tracking"

```
## For usage of package please refer to scripts.
As notebooks are updated frequently and contain more scratch work, hence might create confusion. However, going through the notebook once is recommended for gaining better insight of overall process.
