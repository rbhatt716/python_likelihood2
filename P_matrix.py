#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:32:14 2024

@author: rbhatt
"""

import torch
from scipy.linalg import expm

def P_matrix(Q_list, dt_p,j):
    Q = Q_list[  int(dt_p['row'].iloc[j])   ]
    t = dt_p['interval'].iloc[j]

    p_matrix = expm(Q*t)
     
    return p_matrix


def P_matrix_torch(Q_list, dt_p,dt_p_columns, j):
    unique_cols = list(set(dt_p_columns))
    col_to_idx = {c: i for i, c in enumerate(unique_cols)}

    dt_p_columns_idx = torch.tensor([col_to_idx[c] for c in dt_p_columns])
    dt_p_idx = torch.tensor([col_to_idx[c] for c in ["row"] ])

    mask = torch.isin(dt_p_columns_idx, dt_p_idx)
    Q = Q_list[int(dt_p[j,mask] )]

    dt_p_idx = torch.tensor([col_to_idx[c] for c in ["interval"] ])

    mask = torch.isin(dt_p_columns_idx, dt_p_idx)

    t = dt_p[j,mask]

    p_matrix = torch.matrix_exp(Q * t)

    return p_matrix