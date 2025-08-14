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


def L_i_mats(dta_p,i,Q_list,P_list,death_states,censored_state ,corresponding_censored_states,E_list):
    
    dta_i = dta_p[dta_p['id'] == i].reset_index(drop=True)
    O     = dta_i['state']


    i_len=len(O)

    mats = [mat(Q_list,dta_i,j ,P_list, O, death_states,censored_state ,corresponding_censored_states,E_list   ) for j in range(0,i_len-1)]

    mats_p=reduce(np.matmul, mats)




            
    return    mats_p    
            
            