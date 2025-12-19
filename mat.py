#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:11:34 2024

@author: rbhatt
"""
import numpy as np
import torch

def mat(Q_list,dta_i,j ,P_list, O, death_states,censored_state ,corresponding_censored_states,E_list   ):



    Q = Q_list[  int(dta_i['row'].iloc[j] )   ]
    P=P_list[ int(dta_i['row_p'].iloc[j]) ]

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




    unique_cols = list(set(dt_p_columns))
    col_to_idx = {c: i for i, c in enumerate(unique_cols)}

    dt_p_columns_idx = torch.tensor([col_to_idx[c] for c in dt_p_columns])
    dt_p_idx = torch.tensor([col_to_idx[c] for c in ["row"] ])

    mask = torch.isin(dt_p_columns_idx, dt_p_idx)


def mat_torch(Q_list, dta_p_mats, P_list, death_states, censored_state, corresponding_censored_states, E_list,
        dta_p_mats_columns,j):



    Q = Q_list[int(dta_p_mats[j,dta_p_mats_columns == "row"] ) ]
    P = P_list[int(dta_p_mats[j,dta_p_mats_columns == 'row_p']               ) ]

    no_states = np.shape(Q)[1]

    x = int(dta_p_mats[j,dta_p_mats_columns == "next_state"])

    if x in death_states:
        Q_death = torch.zeros([no_states, no_states],dtype=torch.float64)
        Q_death[:, x - 1] = Q[:, x - 1]
        P_temp = torch.matmul(P, Q_death)
    elif x in censored_state:
        cstates = torch.tensor(corresponding_censored_states[censored_state.index(x)])

        P_temp = torch.zeros([no_states, no_states], dtype=torch.float64)
        P_temp[:, cstates - 1] = P[:, cstates - 1]
    else:
        E = E_list[int(dta_p_mats[j, dta_p_mats_columns == 'next_row_E'])]
        E = torch.diag(E[:, x - 1])
        E=E.to(torch.float64)
        P_temp = torch.matmul(P, E)

    return P_temp
