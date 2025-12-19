
import numpy as np
from hazardfunction import hazards
from hazardfunction import hazards_torch
import torch


def Q_matrix(p,no_qpars,Qm,covariates_name,j,dt,functions,age_like):
    
    indexes = list(range(0,no_qpars,1))
    
    all_qpars = np.array( [p[ind] for ind in indexes])
    

    if len(covariates_name)>0:
        q_pars_mat = all_qpars.reshape(len(covariates_name)+1,  np.sum(Qm>0) )

        cov_vals=np.array(dt[covariates_name].iloc[j])
        cov_names = covariates_name
    else:
        q_pars_mat = all_qpars.reshape(1,  np.sum(Qm>0) )
        cov_vals=[]
        cov_names = []
    
    
    def calculate_qpars(Qm, functions, q_pars_mat, cov_vals, cov_names, age_like):
        qpars = [hazards(functions[i], cov_vals, cov_names, age_like, q_pars_mat[:,i]) for i in range(np.sum(Qm > 0))]
        return qpars
    
    
    qpars = calculate_qpars(Qm, functions, q_pars_mat, cov_vals, cov_names, age_like)
    
    Q = np.zeros((Qm.shape[1],Qm.shape[1]))
    
    k=0
    for i in range(Qm.shape[1]):
        for j in range(Qm.shape[1]):
            if Qm[i,j]>0:
                Q[i,j]=qpars[k]
                k=k+1
    
    for i in range(Qm.shape[1]):
        Q[i,i]=-np.sum(Q,axis=1)[i]
        
        
    return Q


def Q_matrix_torch(p, no_qpars, Qm, covariates_name, j, dt_q,functions,age_like,dt_q_columns):
    indexes = list(range(0, no_qpars, 1))

    all_qpars = p[indexes]

    if len(covariates_name) > 0:
        q_pars_mat = all_qpars.reshape(len(covariates_name) + 1, torch.sum(Qm > 0)  )


        cov_vals =  dt_q[ j, np.isin(dt_q_columns, covariates_name )]
        cov_names = covariates_name
    else:
        q_pars_mat = all_qpars.reshape( 1, torch.sum(Qm > 0)  )
        cov_vals = []
        cov_names = []

    def calculate_qpars(Qm, functions, q_pars_mat, cov_vals, cov_names, age_like):
        qpars = [hazards_torch(functions[i], cov_vals, cov_names, age_like, q_pars_mat[:, i]) for i in range(torch.sum(Qm > 0))]
        return qpars

    qpars = calculate_qpars(Qm, functions, q_pars_mat, cov_vals, cov_names, age_like)

    Q = torch.zeros(Qm.shape[1], Qm.shape[1])

    k = 0
    for i in range(Qm.shape[1]):
        for j in range(Qm.shape[1]):
            if Qm[i, j] > 0:
                Q[i, j] = qpars[k]
                k = k + 1

    for i in range(Qm.shape[1]):
        Q[i, i] = -torch.sum(Q, axis=1)[i]

    return Q