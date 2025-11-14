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
from L_i_lt_mats import L_lt_i_mats
import math 


def L_i_lt(dta_p,i,Q_list,P_list,death_states,censored_state ,corresponding_censored_states,E_list,dta_lt_p, Q_lt_list,P_lt_list,lt_states):
    
    l_i   = L_i_mats(dta_p,i,Q_list,P_list,death_states,censored_state ,corresponding_censored_states,E_list)
    l_i_lt= L_lt_i_mats(dta_lt_p,dta_p,i,Q_lt_list,P_lt_list,death_states,censored_state ,corresponding_censored_states,E_list,lt_states)
    l_i_val = np.matmul(l_i_lt,l_i)


    dta_lt_i = dta_lt_p[dta_lt_p['id'] == i].reset_index(drop=True)
    O     = dta_lt_i['state']
    
    x=O[0] -1
    e_1= np.array([0,0,0])
    e_1[x]= 1
    li = np.sum( np.dot(e_1,l_i_val))
    li=math.log(li)
    
    
    return li    
            
            