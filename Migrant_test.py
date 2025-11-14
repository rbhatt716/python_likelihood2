
sys.path.append('/Users/rbhatt/Library/CloudStorage/OneDrive-Personal/python_likelihood2/python_likelihood2')


from functools import partial
from optimparallel import minimize_parallel
from likelihood_parallelised import loglikelihood
import numpy as np,
import multiprocessing as mp
import pandas as pd
from likelihood_parallelised import objective_top

d = {'id':[1,1,1,2,2,2,3,3,3,4,4,4]   ,'age': [2,2.2,2.4, 3,3.2,3.4, 4,4.2,4.4, 2,2.2,2.5], 'x': [.1,.1,.1 ,  -.3,-.3,-.3,  -.4,-.4,-.4,   .1,.1,.1],'y': [1,1,1, 2,2,2, 3,3,3, 1,1,1],
     'state':[1,2,3, 2,2,3, -1,1,2, 1,-1,2 ]  }
dt = pd.read_csv("/Users/rbhatt/Library/CloudStorage/OneDrive-Personal/South Asian Migrant Project/processed_data.csv")
dt['time'] = dt['age']-34.5

p = [-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194,0,0,0,0,0]
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

p_e=[]
no_epars=0
E_covariates_name=[]
ematrix = np.array([[0,0,0,0],
           [0,0,0,0],
           [0,0,0,0],
           [0, 0, 0, 0]         ]
           )
grid_length = 1
lt_assumption_age = []

p0=np.append(p,p_e)

dt = dt.rename(columns={'QID': 'id'})
dt = dt.rename(columns={'State': 'state'})
dt['id'] = dt['id'].astype(str)


import time
start=time.time()
loglikelihood(p0=p0,dt=dt,covariates_name=covariates_name,no_qpars=no_qpars,no_epars=no_epars,constrain=constrain,Qm=Qm,functions=functions,age_like=age_like,E_covariates_name=E_covariates_name,ematrix=ematrix,Left_trunc=Left_trunc,death_states=death_states,censored_state=censored_state ,corresponding_censored_states=corresponding_censored_states,      lt_states=lt_states,lt_cen_state=lt_cen_state ,use_given_initial_prob=use_given_initial_prob,Use_misclass_matrix=Use_misclass_matrix,initial_prob=initial_prob )
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

