#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:11:34 2024

@author: rbhatt
"""
import numpy as np

def mat(Q_list,dta_i,j ,P_list, O, death_states,censored_state ,corresponding_censored_states,E_list   ):
    Q = Q_list[  dta_i['row'].iloc[j]    ]
    P=P_list[ dta_i['row_p'].iloc[j] ]

    no_states = np.shape(Q)[1]

    x = int(O[j+1])

    if x in death_states:
        Q_death = np.zeros([no_states,no_states])
        Q_death[:, x-1]=Q[:, x-1]
        P_temp = np.matmul(P, Q_death )
    elif x in censored_state:
        cstates =  np.array( corresponding_censored_states[censored_state.index(x)])
        P_temp = np.zeros([no_states,no_states])
        P_temp[:, cstates-1] = P[:, cstates-1]
    else:
        E=E_list[  dta_i['row_E'].iloc[j+1]    ]      
        E = np.diag(E[:, x-1])
        P_temp = np.matmul(P, E )
        
    return P_temp   