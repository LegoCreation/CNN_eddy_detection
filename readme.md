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
$ conda create -n eddy-tracking python=3.8
$ conda activate eddy-tracking

# mamba is faster than conda
$ conda install mamba

# manually install all the dependencies
$ mamba install numpy"<1.21" scipy netCDF4 matplotlib opencv pyyaml pint zarr requests numba">=0.53"

$ pip install polygon3

# Install the algorithm package from the download
$ pip install -e . --no-dependencies

# Create a Kernel for jupyter notebook
$ mamba install ipykernel
$ python -m ipykernel install --user --name eddy-tracking --display-name="eddy-tracking"

```
For usage of package please refer to scripts. As notebooks are updated frequently and contain more scratch work so might create confusion. However, going through the notebook once is recommended for gaining better insight of overall process.
