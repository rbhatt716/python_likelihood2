
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
corresponding_censored_states = [[1,2],[2],[3]]
lt_states = [1,2]
lt_cen_state = -3
constrain = torch.tensor([1,2,3,4,5,6,7,8,9,10 ])

Qm = torch.tensor([[0,0.1,0 ,.1],
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

device = "cuda" if torch.cuda.is_available() else "cpu"
p0 = torch.nn.Parameter(
    torch.tensor(
        [-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194, 0, 0, 0, 0, 0],
        dtype=torch.float32,
        device=device
    )
)


optimizer = torch.optim.SGD([p0], lr=1e-2, momentum=0.9)

for step in range(1000):
    optimizer.zero_grad(set_to_none=True)

    # Your function already returns NEGATIVE loglikelihood (a loss)
    loss = loglikelihood_torch(
        p0=p0,
        covariates_name=covariates_name,
        no_qpars=no_qpars,
        no_epars=no_epars,
        constrain=constrain,
        Qm=Qm,
        functions=functions,
        age_like=age_like,
        E_covariates_name=E_covariates_name,
        ematrix=ematrix,
        death_states=death_states,
        censored_state=censored_state,
        corresponding_censored_states=corresponding_censored_states,
        use_given_initial_prob=use_given_initial_prob,
        Use_misclass_matrix=Use_misclass_matrix,
        initial_prob=initial_prob,
        dt_q=dt_q,
        qlength=qlength,
        plength=plength,
        dt_E=dt_E,
        e_length=e_length,
        dt_q_columns=dt_q_columns,
        dt_E_columns=dt_E_columns,
        dt_p=dt_p,
        dt_p_columns=dt_p_columns,
        dta_p_tensor=dta_p_tensor,
        dta_p_mats=dta_p_mats,
        dta_p_mats_columns=dta_p_mats_columns,
        initial_state=initial_state,
        initial_E=initial_E
    )

    # Backprop + update
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(step, float(loss.detach().cpu()), p0.detach().cpu().numpy())



import time


def loss_only_p0(p0):
    # p0 must be torch Tensor
    return loglikelihood_torch(
        p0=p0,
        covariates_name=covariates_name,
        no_qpars=no_qpars,
        no_epars=no_epars,
        constrain=constrain,
        Qm=Qm,
        functions=functions,
        age_like=age_like,
        E_covariates_name=E_covariates_name,
        ematrix=ematrix,
        death_states=death_states,
        censored_state=censored_state,
        corresponding_censored_states=corresponding_censored_states,
        use_given_initial_prob=use_given_initial_prob,
        Use_misclass_matrix=Use_misclass_matrix,
        initial_prob=initial_prob,
        dt_q=dt_q,
        qlength=qlength,
        plength=plength,
        dt_E=dt_E,
        e_length=e_length,
        dt_q_columns=dt_q_columns,
        dt_E_columns=dt_E_columns,
        dt_p=dt_p,
        dt_p_columns=dt_p_columns,
        dta_p_tensor=dta_p_tensor,
        dta_p_mats=dta_p_mats,
        dta_p_mats_columns=dta_p_mats_columns,
        initial_state=initial_state,
        initial_E=initial_E
    )

# gradcheck needs double precision
p0_test = torch.tensor(
    [-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194, 0, 0, 0, 0, 0],
    dtype=torch.float64,
    requires_grad=True
)

# Run gradcheck
ok = torch.autograd.gradcheck(loss_only_p0, (p0_test,), eps=1e-6, atol=1e-4, rtol=1e-3)
print("gradcheck:", ok)


import torch

def fd_one(f, x, i, eps=1e-6):
    e = torch.zeros_like(x)
    e[i] = 1.0
    return (f(x + eps*e) - f(x - eps*e)) / (2*eps)

# autograd gradient
p = p0_test.clone().detach().requires_grad_(True)
L = loss_only_p0(p)
L.backward()
g_ad = p.grad.detach().clone()

# finite diff gradient at same point
p_det = p0_test.detach().clone()
g_fd = fd_one(loss_only_p0, p_det, i=2, eps=1e-6)

print("AD:", float(g_ad[2]), "FD:", float(g_fd))

for eps in [1e-4, 1e-5, 1e-6, 1e-7]:
    g_fd = fd_one(loss_only_p0, p0_test.detach(), i=2, eps=eps)
    print("eps", eps, "FD grad", float(g_fd))




ok = torch.autograd.gradcheck(
    loss_only_p0,
    (p0_test,),
    eps=1e-4,      # bigger step
    atol=1e-3,
    rtol=1e-2
)
print(ok)

p0 = torch.nn.Parameter(
    torch.tensor(
        [-9.83378713, -6.63901027, -5.80790354, -5.41157498, -2.24235194, 0, 0, 0, 0, 0],
        dtype=torch.float32  # or float32
    )
)

optimizer = torch.optim.SGD([p0], lr=1e-3, momentum=0.9)

for step in range(100):
    optimizer.zero_grad(set_to_none=True)

    loss = loglikelihood_torch(
        p0=p0,
        covariates_name=covariates_name,
        no_qpars=no_qpars,
        no_epars=no_epars,
        constrain=constrain,
        Qm=Qm,
        functions=functions,
        age_like=age_like,
        E_covariates_name=E_covariates_name,
        ematrix=ematrix,
        death_states=death_states,
        censored_state=censored_state,
        corresponding_censored_states=corresponding_censored_states,
        use_given_initial_prob=use_given_initial_prob,
        Use_misclass_matrix=Use_misclass_matrix,
        initial_prob=initial_prob,
        dt_q=dt_q,
        qlength=qlength,
        plength=plength,
        dt_E=dt_E,
        e_length=e_length,
        dt_q_columns=dt_q_columns,
        dt_E_columns=dt_E_columns,
        dt_p=dt_p,
        dt_p_columns=dt_p_columns,
        dta_p_tensor=dta_p_tensor,
        dta_p_mats=dta_p_mats,
        dta_p_mats_columns=dta_p_mats_columns,
        initial_state=initial_state,
        initial_E=initial_E
    )
    loss.backward()

    torch.nn.utils.clip_grad_norm_([p0], max_norm=10.0)
    optimizer.step()

    if step % 5 == 0:
        print(step, float(loss), p0.detach().cpu().numpy())


p = p0.detach().clone().requires_grad_(True)
loss = loss_only_p0(p)

grad = torch.autograd.grad(loss, p, create_graph=True)[0]

H = torch.autograd.functional.hessian(loss_only_p0, p)

H = 0.5 * (H + H.T)