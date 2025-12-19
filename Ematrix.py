#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:25:39 2024

@author: rbhatt
"""

import numpy as np
import torch
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



def E_matrix_torch(p_e, no_epars, ematrix, j, dt_E, E_covariates_name,dt_E_columns):
    indexes = list(range(0, no_epars, 1))
    all_epars = torch.tensor([p_e[ind] for ind in indexes])

    if (E_covariates_name == []):
        epars = all_epars
    else:
        e_pars_mat = all_epars.reshape(len(E_covariates_name) + 1, torch.sum(ematrix > 0))
        Ecov_vals = torch.tensor(dt_E[j, np.isin(dt_E_columns, E_covariates_name)])
        Ecov_vals = torch.cat(( torch.tensor([1]), Ecov_vals))

        epars = Ecov_vals * e_pars_mat

    epars = ([torch.exp(epars[ind]) / (1 + torch.exp(epars[ind])) for ind in range(torch.sum(ematrix > 0))])

    E = torch.zeros((ematrix.shape[1], ematrix.shape[1]))

    k = 0
    for i in range(ematrix.shape[1]):
        for j in range(ematrix.shape[1]):
            if ematrix[i, j] > 0:
                E[i, j] = epars[k]
                k = k + 1

    for i in range(ematrix.shape[1]):
        E[i, i] = 1 - torch.sum(E, axis=1)[i]

    return E
