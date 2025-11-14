#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:32:14 2024

@author: rbhatt
"""

from scipy.linalg import expm

def P_matrix(Q_list, dt_p,j):
    Q = Q_list[  int(dt_p['row'].iloc[j])   ]
    t = dt_p['interval'].iloc[j]

    p_matrix = expm(Q*t)
     
    return p_matrix