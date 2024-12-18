from multiprocessing import Process
import os
import subprocess
import sys

##############################################################################
#                 Produce 12 months of data, in groups of 4                  #
# (Careful! You may use more than 128GB of RAM with more months in parallel) #
##############################################################################

def run_script(config_name, month, script_name="run.py"):
    subprocess.run(["python", script_name, str(month), config_name])

# pass here the .yaml file to use for this year
configuration = sys.argv[1]

# create the first 6 months and then the last 6 to avoid using too much memory
for i in [0, 4, 8]:
    
    processes = []
    for month in range(1+i, 5+i):
        p = Process(target=run_script, args=(configuration, month))
        processes.append(p)
        p.start()
    
    # wait for all processes to finish
    for p in processes:
        p.join()
