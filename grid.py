#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import numpy as np
cwd = os.getcwd() 


# In[ ]:


def unif(prm):
    M = prm[0]
    if M <= 0 or type(M) != type(1):
        print("Number of grid points must be a positive integer!")
        sys.exit(1)
    tail = prm[1]
    x_grid = np.linspace(4, tail, num=M+1, endpoint = True)
    return x_grid

def expt(prm):
    M = prm[0]
    if M <= 0 or type(M) != type(1):
        print("Number of grid points must be a positive integer!")
        sys.exit(1)
    alph = prm[1]
    b = prm[2]

    x_grid = np.array([4 + b * (i**alph) for i in range(M+1)])
    return x_grid

def custom(prm):
    grid_file_path = prm
    try:
        grid = np.load(grid_file_path, allow_pickle=False)
        if not( len(grid.shape) == 1):
            print(f"List of grid points contained in {grid_file_path} must be a 1-dimensional numpy array!")
            sys.exit(1)
        if not( grid.dtype == np.float64 or grid.dtype == np.float128):
            print(f"Grid points contained in {grid_file_path} must be of data type either 'float64' or 'float128' !")
            sys.exit(1)
    except ValueError:
        print(f"{grid_file_path} is an invalid .npy file!")
        sys.exit(1)
    
    if grid[0] > 4:
        x_grid = np.zeros(shape = (len(grid)+1,) , dtype = float)
        x_grid[0] = 4
        x_grid[1:] = grid
        return x_grid
    elif grid[0] < 4:
        print("The grid points must start with 4!")
        sys.exit(1)
    else:
        return grid

def gen(grid_type, prm):
    if grid_type == "uniform":
        return unif(prm)
    elif grid_type == "exp":
        return expt(prm)
    elif grid_type == "custom":
        return custom(prm)
    else:
        print("Grid type must be one of the following values: " + "'uniform', 'exp', 'custom' !")
        sys.exit(1)


# In[ ]:


###jupyter nbconvert --to python grid.ipynb 

