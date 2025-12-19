
sys.path.append('/Users/rbhatt/Library/CloudStorage/OneDrive-Personal/python_likelihood2/python_likelihood2')


from functools import partial
import numpy as np
import pandas as pd
import torch
from likelihood_pytorch import loglikelihood_torch

dt = pd.read_csv("/Users/rbhatt/Library/CloudStorage/OneDrive-Personal/South Asian Migrant Project/processed_data.csv")
dt['time'] = dt['age']-34.5

p = torch.tensor([-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194,0,0,0,0,0])
no_qpars = 10
functions = ['gompertz','gompertz','gompertz','gompertz','gompertz']
age_like=[]
death_states = [4]
censored_state = [-3,-2,-1]
corresponding_censored_states = [[1],[2],[3]]
lt_states = [1,2]
lt_cen_state = -3
constrain = np.array([1,2,3,4,5,6,7,8,9,10 ])

Qm = np.array([[0,0,0.1, .1],
      [0,0,.1,.1],
      [0,0,0,.1],
      [0,0,0,0]
               ])

covariates_name = ['time']
j=0



Use_misclass_matrix=False
use_given_initial_prob=False
initial_prob = []
Left_trunc=False

p_e=torch.tensor([])
no_epars=0
E_covariates_name=[]
ematrix = torch.tensor([[0,0,0,0],
           [0,0,0,0],
           [0,0,0,0],
           [0, 0, 0, 0]         ]
           )
grid_length = 1
lt_assumption_age = []

p0=torch.cat((p.flatten(), p_e.flatten()))

dt = dt.rename(columns={'QID': 'id'})
dt = dt.rename(columns={'State': 'state'})
dt['id'] = dt['id'].astype(str)

if len(covariates_name) > 0:
    dt_q = dt.groupby(covariates_name).size().reset_index(name='counts')
    qlength = len(dt_q.index)
    dt_q['row'] = np.arange(dt_q.shape[0])
else:
    dt_q = dt.assign(counts=len(dt))
    qlength = 1
    dt_q['row'] = 0

dta_s = pd.merge(dt, dt_q, how='left')

dta_s = dta_s.sort_values(by=['id', 'age'])

if len(E_covariates_name) > 0:
    dt_E = dt.groupby(E_covariates_name).size().reset_index(name='counts')

    dt_E['row_E'] = np.arange(dt_E.shape[0])

    e_length = len(dt_E.index)
else:
    dt_E = {'row_E': 0}
    e_length = 1

if len(E_covariates_name) > 0:

    dta_s = pd.merge(dta_s, dt_E, how='left')
else:
    dta_s['row_E'] = 0

dta_s = dta_s.sort_values(by=['id', 'age'])

dt_q_columns = dt_q.columns
dt_q = dt_q.to_numpy()
dt_q = torch.from_numpy(dt_q)

if len(E_covariates_name) > 0:
    dt_E_columns = dt_E.columns
    dt_E = dt_E.to_numpy()
    dt_E = torch.from_numpy(dt_E)
else:
    dt_E_columns = []

dta_p = dta_s
dta_p['lead_time'] = dta_s.groupby(['id'])['age'].shift(-1)
dta_p['interval'] = dta_p['lead_time'] - dta_p['age']

dta_p = dta_p.fillna(0)

dt_p = dta_p.groupby(['row', 'interval']).size().reset_index(name='counts')

dt_p['row_p'] = np.arange(dt_p.shape[0])




plength = len(dt_p.index)


dta_p = dta_p.drop(columns=['counts'])
dta_p = pd.merge(dta_p, dt_p, how='left')
dta_p = dta_p.sort_values(by=['id', 'age'])

dta_p["next_state"] = (
    dta_p
    .groupby("id")["state"]
    .shift(-1)
)


dta_p["next_row_E"] = (
    dta_p
    .groupby("id")["row_E"]
    .shift(-1)
)

dt_p_columns = dt_p.columns
dt_p = dt_p.to_numpy()
dt_p = torch.from_numpy(dt_p)


dta_p_mats = dta_p.groupby(["state", "row","row_E" ,'row_p',"next_state","next_row_E" ]).size().reset_index(name='counts')






dta_p_mats['row_m'] = np.arange(int(dta_p_mats.shape[0]))

dta_p = dta_p.drop(columns=['counts'])

dta_p = pd.merge(dta_p, dta_p_mats, how='left')



dta_p_mats_columns = dta_p_mats.columns
dta_p_mats = dta_p_mats.to_numpy()
dta_p_mats = torch.from_numpy(dta_p_mats)


dta_p_grouped = dta_p.dropna()

dta_p_grouped = dta_p_grouped.groupby("id")["row_m"].apply(list)

nested_list = dta_p_grouped.tolist()

dta_p_tensor = [torch.tensor(x,dtype=torch.int64) for x in nested_list]

initial_state = dta_p.loc[dt.groupby('id')['age'].idxmin()]
initial_state = initial_state['state'].to_numpy()
initial_state = torch.from_numpy(initial_state)

initial_E = dta_p.loc[dt.groupby('id')['age'].idxmin()]
initial_E = initial_E['row_E'].to_numpy()
initial_E = torch.from_numpy(initial_E)


import time
start=time.time()
loglikelihood_torch(p0=p0,covariates_name=covariates_name,no_qpars=no_qpars,no_epars=no_epars,constrain=constrain,Qm=Qm,functions=functions,age_like=age_like,E_covariates_name=E_covariates_name,ematrix=ematrix,death_states=death_states,censored_state=censored_state ,corresponding_censored_states=corresponding_censored_states, use_given_initial_prob=use_given_initial_prob,Use_misclass_matrix=Use_misclass_matrix,initial_prob=initial_prob,dt_q=dt_q, qlength=qlength,plength=plength,dt_E=dt_E,e_length=e_length,dt_q_columns=dt_q_columns,dt_E_columns=dt_E_columns,dt_p=dt_p, dt_p_columns=dt_p_columns,dta_p_tensor=dta_p_tensor,dta_p_mats=dta_p_mats,dta_p_mats_columns=dta_p_mats_columns,initial_state=initial_state,initial_E=initial_E )
end=time.time()

fixed = dict( dt=dt, covariates_name=covariates_name, no_qpars=no_qpars, no_epars=no_epars,
              constrain=constrain, Qm=Qm, functions=functions, age_like=age_like,
              E_covariates_name=E_covariates_name, ematrix=ematrix, Left_trunc=Left_trunc,
              death_states=death_states, censored_state=censored_state,
              corresponding_censored_states=corresponding_censored_states,
              lt_states=lt_states, lt_cen_state=lt_cen_state,
              use_given_initial_prob=use_given_initial_prob,
              Use_misclass_matrix=Use_misclass_matrix, initial_prob=initial_prob )

p0 = np.array([-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194, 0,0,0,0,0], float)
bounds = [(-10, 10)] * p0.size  # or clip p0 into (-5,5) if those are the real bounds

objective = partial(objective_top, fixed=fixed)
start=time.time()

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    res = minimize_parallel(fun=objective, x0=p0, bounds=bounds)
    print(res)
end=time.time()

