#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:00:55 2024

@author: rbhatt
"""

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


def L_lt_i_mats(dta_lt_p,dta_p,i,Q_lt_list,P_lt_list,death_states,censored_state ,corresponding_censored_states,E_list,lt_states):
    
    dta_lt_i = dta_lt_p[dta_lt_p['id'] == i].reset_index(drop=True)
    O     = dta_lt_i['state']
    dta_i = dta_p[dta_p['id'] == i].reset_index(drop=True)
    
    row_E=dta_i['row_E'][0]
    emat = E_list[row_E]
    
    i_len=len(O)
    
    dta_lt_i['row_E']=0

    dta_lt_i.loc[i_len-1, "row_E"] =1

    no_states = np.shape(Q_lt_list[0])[1]
    E_lt_list= np.array([np.identity(no_states),emat])

    mats = [mat(Q_lt_list,dta_lt_i,j ,P_lt_list, O, death_states,censored_state ,corresponding_censored_states, E_lt_list ) for j in range(0,i_len-1)]

    mats_p=np.array(reduce(np.matmul, mats))

    p_val=np.sum( mats_p[(O[0]-1),(np.array(lt_states)-1)] )  



    mats_p[~np.isin(np.arange(mats_p.shape[0]), O[0]-1 ), :]=0
    mats_p[ :,~np.isin(np.arange(mats_p.shape[1]), (np.array(lt_states)-1)) ]=0

    mats_p=mats_p/p_val


    return    mats_p    
            
            