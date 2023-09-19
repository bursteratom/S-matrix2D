#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import numpy as np
cwd = os.getcwd()


# In[2]:


def validate_config(config):
    def _is_int_(number_string):
        try:
            number = int(number_string)
            return True
        except ValueError:
            return False
    
    def _is_float_(number_string):
        try:
            number = float(number_string)
            return True
        except ValueError:
            return False

    keys = ['MASS of LIGHTEST PARTICLE', 'MASS SPECTRUM', 'GRID TYPE', 'GRID PARAMETRE(S)']
    for key in keys:
        if key in config.keys():
            continue
        else:
            print(f"Missing parameter: '{key}' ")
            sys.exit(1)
    
    config['MASS of LIGHTEST PARTICLE'] = config['MASS of LIGHTEST PARTICLE'].split()[0]
    config['GRID TYPE'] = config['GRID TYPE'].split()[0]

    if not _is_float_(config['MASS of LIGHTEST PARTICLE']) :
        print(f"Value of parametre 'MASS of LIGHTEST PARTICLE': '{config['MASS of LIGHTEST PARTICLE']}' is invalid!")
        sys.exit(1)

    for s in config['MASS SPECTRUM'].split():
        is_num = _is_float_(s)
        if not is_num :
            print(f"One of MASS SPECTRUM values, '{s}', is invalid!")
            sys.exit(1)
    
    if config['GRID TYPE'].lower() == 'uniform':
        if not( len(config['GRID PARAMETRE(S)'].split() ) == 2 ):
            print("Uniform('uniform') grid requires two and only two parametres!")
            sys.exit(1)
        else:
            if not(_is_int_(config['GRID PARAMETRE(S)'].split()[0])):
                print(f"Number of grid points (first argument) in 'GRID PARAMETRE(S)' must be a positive integer!")
                sys.exit(1)
            elif int(config['GRID PARAMETRE(S)'].split()[0]) < 0:
                print(f"Number of grid points (first argument) in 'GRID PARAMETRE(S)' must be a positive integer!")
                sys.exit(1)
            for s in config['GRID PARAMETRE(S)'].split():
                is_num = _is_float_(s)
                if not is_num :
                    print(f"One of 'GRID PARAMETRE(S)' values, {s}, is invalid!")
                    sys.exit(1)
                if not(float(s) >0 ):
                    print("'GRID PARAMETRE(S)' values must be greater than 0!")
                    sys.exit(1)
            if not( float(config['GRID PARAMETRE(S)'].split()[1]) > 4 ):
                print(f"Grid tail (second argument) in 'GRID PARAMETRE(S)' must be greater than 4!")
                sys.exit(1)
    
    elif config['GRID TYPE'].lower() == 'exp':
        if not( len(config['GRID PARAMETRE(S)'].split() ) == 3 ):
            print("Exponential('exp') grid requires three and only three parametres!")
            sys.exit(1)
        else:
            if not(_is_int_(config['GRID PARAMETRE(S)'].split()[0])):
                print(f"Number of grid points (first argument) in 'GRID PARAMETRE(S)' must be a positive integer!")
                sys.exit(1)
            elif int(config['GRID PARAMETRE(S)'].split()[0]) < 0:
                print(f"Number of grid points (first argument) in 'GRID PARAMETRE(S)' must be a positive integer!")
                sys.exit(1)
            for s in config['GRID PARAMETRE(S)'].split():
                is_num = _is_float_(s)
                if not is_num :
                    print(f"One of 'GRID PARAMETRE(S)' values, {s}, is invalid!")
                    sys.exit(1)
                if not(float(s) > 0 ):
                    print("'GRID PARAMETRE(S)' values must be greater than 0!")
                    sys.exit(1)
            if not _is_int_(config['GRID PARAMETRE(S)'].split()[2] )  :
                print(r"Number of grid points (third argument) in 'GRID PARAMETRE(S)' must be an integer greater than 0!")
                sys.exit(1)

    elif config['GRID TYPE'].lower() == 'custom':
        config['GRID PARAMETRE(S)'] = config['GRID PARAMETRE(S)'].split()[0]
        read_dir = os.path.join(cwd, "init_config")
        grid_file = config['GRID PARAMETRE(S)']
        grid_file_path = os.path.join(read_dir, grid_file)
        if not(os.path.exists(grid_file_path)):
            print(f"Grid file {grid_file_path} does not exist!")
            sys.exit(1)
        
        try:
            grid = np.load(grid_file_path, allow_pickle=False)
            if not( len(grid.shape) == 1):
                print(f"List of grid points contained in {grid_file_path} must be a 1-dimensional numpy array!")
                sys.exit(1)
            if not( grid.dtype == np.float64 or grid.dtype == np.float128):
                print(f"Grid points contained in {grid_file_path} must be of data type either 'float64' or 'float128' !")
                sys.exit(1)
            config['GRID PARAMETRE(S)'] = grid_file_path
        except ValueError:
            print(f"{grid_file_path} is an invalid .npy file!")
            sys.exit(1)
        
    else:
        print("Paramtre 'GRID TYPE' must be one of the following values: " + "'uniform', 'exp', 'custom' !")
        sys.exit(1)
    
    return config
      

def read_config():
    def _conv_params_(params):
        params_conved = {}
        keys = ['MASS of LIGHTEST PARTICLE', 'MASS SPECTRUM', 'GRID TYPE', 'GRID PARAMETRE(S)']
        params_conved['MASS of LIGHTEST PARTICLE'] = float(params['MASS of LIGHTEST PARTICLE'])
        params_conved['MASS SPECTRUM'] = [ float(m_i) for m_i in params['MASS SPECTRUM'].split() ]
        params_conved['GRID TYPE'] = params['GRID TYPE']
        if params['GRID TYPE'] == "uniform" or params['GRID TYPE'] == "exp":
            params_conved['GRID PARAMETRE(S)'] = [ int(params['GRID PARAMETRE(S)'].split()[0]) ]
            for aa in range(1, len(params['GRID PARAMETRE(S)'].split())):
                params_conved['GRID PARAMETRE(S)'].append( float(params['GRID PARAMETRE(S)'].split()[aa]) )
        elif params['GRID TYPE'] == "custom":
            params_conved['GRID PARAMETRE(S)'] = params['GRID PARAMETRE(S)']
        else:
            print("Paramtre 'GRID TYPE' must be one of the following values: " + "'uniform', 'exp', 'custom' !")
            sys.exit(1)

        return params_conved

    read_dir = os.path.join(cwd, "init_config")
    config_file = "config.txt"
    read_file = os.path.join(read_dir, config_file)

    with open(read_file,'r') as f:
        lines = f.readlines()
        f.close()
    
    params = {}
    for line in lines:
        #print(line) ## for debugging purpose 
        line_copy = line.split('#')[0] 
        if line_copy == '': ## remove comments
            continue
        [key, val] = line_copy.split(':')
        params[key] = val.lower()
    
    params = validate_config(params)
    params = _conv_params_(params)

    return params



# In[3]:


##jupyter nbconvert --to python config.ipynb 

