#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:35:27 2024

@author: rbhatt

"""
import numpy as np
from mat import mat
from Ematrix import E_matrix
from functools import reduce
from L_i_mats import L_i_mats
import math 


def L_i(dta_p,i,Q_list,P_list,death_states,censored_state ,corresponding_censored_states,E_list,use_given_initial_prob,Use_misclass_matrix,p_e,no_epars,ematrix,E_covariates_name,initial_prob ):
    
    l_i   = L_i_mats(dta_p,i,Q_list,P_list,death_states,censored_state ,corresponding_censored_states,E_list)

    dta_i = dta_p[dta_p['id'] == i].reset_index(drop=True)
    O     = dta_i['state']
    
    if use_given_initial_prob==False and Use_misclass_matrix==False:
               x=O[0] -1
               e_1= np.array([0,0,0])
               e_1[x]= 1
               li = np.sum( np.dot(e_1,l_i))
              
    elif use_given_initial_prob==True and Use_misclass_matrix==False:
                x=O[0] -1
                e_1= np.array([1,0,0]) 
                e_1[x] = np.array(initial_prob[x])
                e_1= np.array(initial_prob[x])
                li = np.sum( np.dot(e_1,l_i[x,:]))
    else:
                dta_0 = dta_i.loc[: , E_covariates_name].reset_index( drop=True)
                E_0=E_matrix(p_e,no_epars,ematrix,0,dta_0,E_covariates_name) 
                x=O[0] -1
                e_1=  np.array(E_0[x,])                
                li = np.sum(np.matmul(  np.diag(e_1),l_i))
                
    li = math.log(li)
    
    return li    
            
            