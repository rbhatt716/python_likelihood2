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
import pandas as pd
import math
import os
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

# ------------------------------------------------------------------
# 0) Your functions as-is
# ------------------------------------------------------------------

def L_i(dta_p, i, Q_list, P_list, death_states, censored_state,
        corresponding_censored_states, E_list, use_given_initial_prob,
        Use_misclass_matrix, p_e, no_epars, ematrix,
        E_covariates_name, initial_prob):

    l_i = L_i_mats(dta_p, i, Q_list, P_list, death_states,
                   censored_state, corresponding_censored_states, E_list)

    dta_i = dta_p[dta_p['id'] == i].reset_index(drop=True)
    O = dta_i['state']

    if use_given_initial_prob == False and Use_misclass_matrix == False:
        x = int(O[0]) - 1
        e_1 = np.zeros(l_i.shape[1])
        e_1[x] = 1
        li = np.sum(np.dot(e_1, l_i))

    elif use_given_initial_prob == True and Use_misclass_matrix == False:
        x = int(O[0]) - 1
        e_1 = np.zeros(l_i.shape[1])
        e_1[x] = np.array(initial_prob[x])
        e_1 = np.array(initial_prob[x])
        li = np.sum(np.dot(e_1, l_i[x, :]))

    else:
        dta_0 = dta_i.loc[:, E_covariates_name].reset_index(drop=True)
        E_0 = E_matrix(p_e, no_epars, ematrix, 0, dta_0, E_covariates_name)
        x = int(O[0]) - 1
        e_1 = np.array(E_0[x, ])
        li = np.sum(np.matmul(np.diag(e_1), l_i))

    li = math.log(li)
    return li


# ------------------------------------------------------------------
# 1) Helpers to create/attach shared memory blocks for 3D arrays
#    Q_list: (nQ, 4, 4), P_list: (nP, 4, 4)
# ------------------------------------------------------------------

def _stack_mats(mat_list):
    """Ensure a contiguous float64 array of shape (N, 4, 4)."""
    arr = np.asarray(mat_list, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (4, 4):
        arr = arr[None, :, :]
    assert arr.ndim == 3 and arr.shape[1:] == (4, 4), "Matrices must be 4x4."
    return np.ascontiguousarray(arr)

def _create_shm_for_array(arr):
    """Create a shared memory block and copy arr into it. Returns (name, shape, dtype)."""
    shm = SharedMemory(create=True, size=arr.nbytes)
    # Write data
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    # Return metadata so workers can re-attach
    return shm, {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}

def _attach_shm_array(meta):
    """Attach to an existing shared memory block using metadata."""
    shm = SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    return shm, arr


# ------------------------------------------------------------------
# 2) Initializer: attach to shared memory & stash everything in globals
# ------------------------------------------------------------------

def _init_worker(_dta_p,
                 _Q_meta, _P_meta,
                 _death_states, _censored_state, _corresponding_censored_states,
                 _E_list, _use_given_initial_prob, _Use_misclass_matrix,
                 _p_e, _no_epars, _ematrix, _E_covariates_name, _initial_prob):

    global dta_p_g, Q_list_g, P_list_g, death_states_g, censored_state_g
    global corresponding_censored_states_g, E_list_g, use_given_initial_prob_g
    global Use_misclass_matrix_g, p_e_g, no_epars_g, ematrix_g
    global E_covariates_name_g, initial_prob_g
    global Q_shm_g, P_shm_g   # keep handles so workers can close on exit

    # Regular big Python objects (sent once per worker, not per task)
    dta_p_g = _dta_p
    death_states_g = _death_states
    censored_state_g = _censored_state
    corresponding_censored_states_g = _corresponding_censored_states
    E_list_g = _E_list
    use_given_initial_prob_g = _use_given_initial_prob
    Use_misclass_matrix_g = _Use_misclass_matrix
    p_e_g = _p_e
    no_epars_g = _no_epars
    ematrix_g = _ematrix
    E_covariates_name_g = _E_covariates_name
    initial_prob_g = _initial_prob

    # Attach to shared memory for Q and P (zero-copy across workers)
    Q_shm_g, Q_arr = _attach_shm_array(_Q_meta)
    P_shm_g, P_arr = _attach_shm_array(_P_meta)

    # Present them to your existing code as lists of 4x4 views
    # (list of views—no data copies)
    Q_list_g = [Q_arr[i] for i in range(Q_arr.shape[0])]
    P_list_g = [P_arr[i] for i in range(P_arr.shape[0])]


def _worker(i):
    return L_i(
        dta_p_g, i, Q_list_g, P_list_g, death_states_g, censored_state_g,
        corresponding_censored_states_g, E_list_g, use_given_initial_prob_g,
        Use_misclass_matrix_g, p_e_g, no_epars_g, ematrix_g,
        E_covariates_name_g, initial_prob_g
    )


# ------------------------------------------------------------------
# 3) Driver: prepares shared memory, runs pool, cleans up properly
# ------------------------------------------------------------------

# Prevent BLAS oversubscription (massive win for many 4x4 ops)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def compute_Li_parallel_shared(
    dta_p, Q_list, P_list, death_states, censored_state,
    corresponding_censored_states, E_list, use_given_initial_prob,
    Use_misclass_matrix, p_e, no_epars, ematrix, E_covariates_name, initial_prob,
    processes=None, chunksize=200
):
    # 1) Pack Q_list / P_list into contiguous arrays
    Q_arr = _stack_mats(Q_list)      # shape (nQ, 4, 4)
    P_arr = _stack_mats(P_list)      # shape (nP, 4, 4)

    # 2) Create shared memory blocks (parent owns them)
    Q_shm, Q_meta = _create_shm_for_array(Q_arr)
    P_shm, P_meta = _create_shm_for_array(P_arr)

    try:
        # 3) Launch worker pool; only ids go through the pipe at runtime
        ids = pd.unique(dta_p["id"]).tolist()

        with get_context("spawn").Pool(
            processes=processes,
            initializer=_init_worker,
            initargs=(
                dta_p,                  # pandas DF (sent once per worker)
                Q_meta, P_meta,         # shared memory metadata (tiny)
                death_states, censored_state, corresponding_censored_states,
                E_list, use_given_initial_prob, Use_misclass_matrix,
                p_e, no_epars, ematrix, E_covariates_name, initial_prob
            ),
        ) as pool:
            results = pool.map(_worker, ids, chunksize=chunksize)

    finally:
        # 4) Clean up shared memory in parent
        #    (Workers automatically close their handles when they exit.)
        try:
            Q_shm.close(); Q_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            P_shm.close(); P_shm.unlink()
        except FileNotFoundError:
            pass

    return results