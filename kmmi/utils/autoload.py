import os
from pathlib import Path
import pickle
import hashlib

def autoload(fname, func):
    """Function for autoloading / saving costly variables to HDD. 
    """
    fname = 'tempvars/{}.pickle.tmp'.format(fname)
    # Variable is stored in hdd, load
    if os.path.exists(fname):
        var = __load_var(fname) 
    
    # Variable is not present, exec func and save
    else:
        var = func()
        __save_var(var, fname)
    
    return var

def __load_var(fname):
    """Helper routine for loading temporary / computationally
    expensive results from hdd."""
    
    with open(fname, 'rb') as file:
        return pickle.load(file)

def __save_var(var, fname):
    """Helper routine for saving temporary / computationally
    expensive results to hdd."""
    
    # Create tempvars directory
    Path("tempvars").mkdir(parents=True, exist_ok=True)
    
    with open(fname, 'wb') as file:
        pickle.dump(var, file)