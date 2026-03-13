import pandas as pd 
import numpy as np
from gurobipy import Model, GRB, quicksum
from src.optimization.utils import haversine

def bike_rebalancing(stations,
                     pred_df,
                     hour_to_optimize,
                     candidate_stations,
                     cluster_nb,
                     gamma,
                     nb_trucks,
                     random_init_state=False,
                     opt_MIPGap:float=0.05,
                     opt_time_limit:int=60):
    
    hto = pred_df[pred_df['hour']==hour_to_optimize]
    if random_init_state == True : 
        np.random.seed(8)
        hto['current_bike_nb'] = (np.clip(
                np.random.normal(0.5 * hto['dpcapacity'], 0.2 * hto['dpcapacity']),
                0,
                hto['dpcapacity']
            )).astype(int)
    selected_station_ids = candidate_stations[candidate_stations['cluster']==cluster_nb]['id'].to_list()
    selected_station_ids = selected_station_ids + [0]
    w_stations = stations[stations['id'].isin(selected_station_ids)]
    w_hto = hto[hto['station_id'].isin(selected_station_ids)]

    # Sets
    I_all = w_stations['id'].to_list() # all stations id + warehouse 
    I = w_stations['id'][:-1].to_list() # all stations id without warehouse
    K = range(nb_trucks) # all trucks

    # Parameters
    alpha = {i: 0.2 for i in I} # service-level requirement
    C = {i: w_stations.loc[w_stations['id'] == i, 'dpcapacity'].values[0] for i in I} # capacity

    S = {i: w_hto.loc[w_hto['station_id'] == i, 'current_bike_nb'].values[0] for i in I}

    delta = {i: w_hto.loc[w_hto['station_id'] == i, 'pred_net_demand'].values[0] for i in I}  # predicted demand
    D = {i: {} for i in I_all}     # distance matrix D[i][j]
    for i in I_all :
        lat_i = w_stations.loc[w_stations['id'] == i, 'latitude'].values[0]
        lon_i = w_stations.loc[w_stations['id'] == i, 'longitude'].values[0]
        for j in I_all : 
            lat_j = w_stations.loc[w_stations['id'] == j, 'latitude'].values[0]
            lon_j = w_stations.loc[w_stations['id'] == j, 'longitude'].values[0]
            D[i][j] = haversine(lat_i, lon_i, lat_j, lon_j)
    M = 200
    Tmax = {k: 30 for k in K} # truck capacities
    warehouse_id = 0

    ######################
    # Model init
    model = Model("BikeRebalancing")

    # Decision Variables
    X = model.addVars(I, K, vtype=GRB.CONTINUOUS, name="X") # bikes drop off by truck k at station i
    Y = model.addVars(I, K, vtype=GRB.CONTINUOUS, name="Y") # bikes picked up by truck k at station i
    W = model.addVars(I_all, I_all, K, vtype=GRB.BINARY, name="W") # trip between ith and jth station
    T = model.addVars(I_all, K, vtype=GRB.CONTINUOUS, name="T") # nb of bikes before entering station i
    P = model.addVars(I, vtype=GRB.CONTINUOUS, name="P") # bike penalty associated with station i
    Q = model.addVars(I, vtype=GRB.CONTINUOUS, name="Q") # dock penalty associated with station i
    y = model.addVars(I, vtype=GRB.BINARY, name="y")
    z = model.addVars(I, vtype=GRB.BINARY, name="z")
    U = model.addVars(I_all, K, vtype=GRB.INTEGER, name="U") # visit order of truck k 

    ###################
    # objective 
    model.setObjective(
        quicksum(W[i,j,k]*D[i][j] for i in I_all for j in I_all for k in K) + 
        gamma * quicksum(P[i]+Q[i] for i in I),
        GRB.MINIMIZE
    )
    ###################
    # Constraints

    # Non-negativity (maybe not necessary)

    for i in I:
        for k in K:
            model.addConstr(X[i,k] >= 0)
            model.addConstr(Y[i,k] >= 0)

    for i in I_all:
        for k in K:
            model.addConstr(T[i,k] >= 0)

    # Visit order bounds

    for i in I:
        for k in K:
            model.addConstr(U[i,k] >= 1)
            model.addConstr(U[i,k] <= len(I))
    for k in K:
        model.addConstr(U[warehouse_id,k] == 0)


    # Stable system
    model.addConstr(quicksum(X[i,k] for i in I for k in K) == quicksum(Y[i,k] for i in I for k in K))

    # Station capacity
    for i in I:
        model.addConstr(S[i] + quicksum(X[i,k] for k in K) - quicksum(Y[i,k] for k in K) <= C[i])
        model.addConstr(S[i] + quicksum(X[i,k] for k in K) - quicksum(Y[i,k] for k in K) >= 0)

    # Truck routing logic

    for i in I:
        for k in K:
            model.addConstr(quicksum(W[i,j,k] for j in I_all) <= X[i,k] + Y[i,k])
            model.addConstr(X[i,k] + Y[i,k] <= M * quicksum(W[i,j,k] for j in I_all))

    for i in I_all:
        for k in K:
            model.addConstr(quicksum(W[i,j,k] for j in I_all) <= 1)
            model.addConstr(quicksum(W[i,j,k] for j in I_all) == quicksum(W[j,i,k] for j in I_all))
            model.addConstr(W[i,i,k] == 0)
    for k in K:
        model.addConstr(quicksum(W[warehouse_id,j,k] for j in I_all) == 1)

    # Truck bike-storing logic
    for i in I_all:
        for k in K:
            model.addConstr(T[i,k] <= Tmax[k])
            model.addConstr(T[i,k] <= M * (1 - W[warehouse_id,i,k]))

    for i in I : 
        for k in K:
            model.addConstr(X[i,k] <= T[i,k])
            model.addConstr(Y[i,k] <= Tmax[k] - T[i,k])

    for i in I:
        for j in I_all:
            if i != j:
                for k in K:
                    model.addConstr(T[j,k] <= T[i,k] + Y[i,k] - X[i,k] + M*(1 - W[i,j,k]))
                    model.addConstr(T[j,k] >= T[i,k] + Y[i,k] - X[i,k] - M*(1 - W[i,j,k]))
                
    for j in I_all:
        if 0 != j:
            for k in K:
                model.addConstr(T[j,k] <= T[0,k] + M*(1 - W[0,j,k]))
                model.addConstr(T[j,k] >= T[0,k] - M*(1 - W[0,j,k]))
    for k in K:
        model.addConstr(T[warehouse_id,k] == 0)

    # Service-level constraints (exclude depot)
    for i in I:
        model.addConstr(alpha[i]*C[i] - (S[i] - delta[i] + sum(X[i,k] for k in K) - sum(Y[i,k] for k in K)) <= P[i])
        model.addConstr(P[i] <= M * y[i])
        model.addConstr(alpha[i]*C[i] - (S[i] - delta[i] + sum(X[i,k] for k in K) - sum(Y[i,k] for k in K)) >= P[i] - M*(1-y[i]))
        model.addConstr((alpha[i]-1)*C[i] + (S[i] - delta[i] + sum(X[i,k] for k in K) - sum(Y[i,k] for k in K)) <= Q[i])
        model.addConstr(Q[i] <= M * z[i])
        model.addConstr((alpha[i]-1)*C[i] + (S[i] - delta[i] + sum(X[i,k] for k in K) - sum(Y[i,k] for k in K)) >= Q[i] - M*(1-z[i]))


    # MTZ subtour elimination (exclude depot)
    for i in I:
        for j in I:
            if i != j:
                for k in K:
                    model.addConstr(U[i,k] - U[j,k] + len(I) * W[i,j,k] <= len(I)-1)
                

    model.update()
    #model.printStats()
    model.setParam("OutputFlag", 0)
    model.Params.MIPGap = opt_MIPGap
    model.Params.TimeLimit = opt_time_limit
    model.optimize()


# Check feasibility 
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
        
        solution = {
            "X": {(i,k): X[i,k].X for i in I for k in K},
            "Y": {(i,k): Y[i,k].X for i in I for k in K},
            "T": {(i,k): T[i,k].X for i in I_all for k in K},
            "W": {(i,j,k): W[i,j,k].X for i in I_all for j in I_all for k in K},
            "P": {i: P[i].X for i in I},
            "Q": {i: Q[i].X for i in I},
            "y": {i: y[i].X for i in I},
            "z": {i: z[i].X for i in I},
            "U": {(i,k): U[i,k].X for i in I_all for k in K},
            "objective": model.ObjVal,
            "D": D,
            "I": I,
            "I_all": I_all,
            "K": list(K),
            "warehouse_id": warehouse_id,
            "S": S,
            "delta": delta,
            "C": C,
            "Tmax": Tmax
        }
    else:
        
        solution = {
            "X": {(i,k): 0 for i in I for k in K},
            "Y": {(i,k): 0 for i in I for k in K},
            "T": {(i,k): 0 for i in I_all for k in K},
            "W": {(i,j,k): 0 for i in I_all for j in I_all for k in K},
            "P": {i: 0 for i in I},
            "Q": {i: 0 for i in I},
            "y": {i: 0 for i in I},
            "z": {i: 0 for i in I},
            "U": {(i,k): 0 for i in I_all for k in K},
            "objective": None,
            "D": D,
            "I": I,
            "I_all": I_all,
            "K": list(K),
            "warehouse_id": warehouse_id,
            "S": S,
            "delta": delta,
            "C": C,
            "Tmax": Tmax
        }
    status_dict = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "Infeasible or Unbounded",
        GRB.TIME_LIMIT: "Time limit reached",
        GRB.SUBOPTIMAL: "Suboptimal"
    }
    print("Solver status:", status_dict.get(model.status, model.status))

    return solution