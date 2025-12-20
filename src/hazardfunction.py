#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:47:38 2024

@author: rbhatt
"""

import numpy as np
import math 

import torch


def hazards(fname,cov_vals,cov_names,age_like,params ):

  if(fname == "gompertz"):
    
    
     
    temp_covals = np.concatenate(([1], cov_vals))
    
    log_gmps = sum(temp_covals*params)
    
    return  math.exp(log_gmps)
    
    
  elif (fname=="exponential"):
    
    
    temp_covals = np.concatenate(([1], cov_vals))

    log_exp = sum(temp_covals*params)
    return  math.exp(log_exp)
    
    

  


def hazards_torch(fname, cov_vals, cov_names, age_like, params):
    """
    fname      : "gompertz" or "exponential"
    cov_vals   : 1D tensor (covariate values, without intercept)
    params     : 1D tensor (same length as [1] + cov_vals)
    cov_names  : unused here but kept for interface compatibility
    age_like   : unused here but kept for interface compatibility
    """

    # Ensure everything is a torch tensor on the same device / dtype
    cov_vals = torch.as_tensor(cov_vals, dtype=params.dtype, device=params.device)
    params   = torch.as_tensor(params,   dtype=cov_vals.dtype, device=cov_vals.device)

    # Add intercept = 1 at the front: [1, x1, x2, ...]
    intercept = torch.ones(1, dtype=cov_vals.dtype, device=cov_vals.device)
    temp_covals = torch.cat((intercept, cov_vals))   # shape: (p,)

    # Linear predictor
    linpred = torch.dot(temp_covals, params)         # scalar tensor

    if fname == "gompertz":
        return torch.exp(linpred)                    # hazard > 0
    elif fname == "exponential":
        return torch.exp(linpred)
    else:
        raise ValueError(f"Unknown hazard function: {fname}")
