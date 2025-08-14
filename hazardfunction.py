#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:47:38 2024

@author: rbhatt
"""

import numpy as np
import math 



def hazards(fname,cov_vals,cov_names,age_like,params ):

  if(fname == "gompertz"):
    
    
     
    temp_covals = np.concatenate(([1], cov_vals))
    
    log_gmps = sum(temp_covals*params)
    
    return  math.exp(log_gmps)
    
    
  elif (fname=="exponential"):
    
    
    temp_covals = np.concatenate(([1], cov_vals))

    log_exp = sum(temp_covals*params)
    return  math.exp(log_exp)
    
    

  

