#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:25:39 2024

@author: rbhatt
"""

import numpy as np
import pandas as pd
import math



def E_matrix(p_e,no_epars,ematrix,j,dt_E,E_covariates_name):

    indexes = list(range(0,no_epars,1))
    all_epars = np.array( [p_e[ind] for ind in indexes])

    if(E_covariates_name == []  ):
        epars = all_epars
    else:
        e_pars_mat = all_epars.reshape(len(E_covariates_name)+1,  np.sum(ematrix>0) )
        Ecov_vals=np.array(dt_E[E_covariates_name].iloc[j])
        Ecov_vals=np.concatenate(([1], Ecov_vals))
        epars =np.dot(Ecov_vals,e_pars_mat)

    epars=np.array( [math.exp(epars[ind])/(1+math.exp(epars[ind] )) for ind in range(np.sum(ematrix>0)) ])


    E = np.zeros((ematrix.shape[1],ematrix.shape[1]))
    
    k=0
    for i in range(ematrix.shape[1]):
        for j in range(ematrix.shape[1]):
            if ematrix[i,j]>0:
                E[i,j]=epars[k]
                k=k+1
                
    
    for i in range(ematrix.shape[1]):
        E[i,i]=1-np.sum(E,axis=1)[i]
        
        
    return E           


