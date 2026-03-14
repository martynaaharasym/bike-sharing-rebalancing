import numpy as np
import pandas as pd
import time
from src.optimization.clustering import cluster_candidate_station
from src.optimization.bike_rebalancing import bike_rebalancing

def simulate_bike_day(df, day):

    day = pd.to_datetime(day).normalize()

    if not pd.api.types.is_datetime64_any_dtype(df["hour"]):
        raise ValueError("Column 'hour' must be of datetime type.")

    min_date = df["hour"].min().normalize()
    max_date = df["hour"].max().normalize()

    if not (min_date <= day <= max_date):
        raise ValueError(
            f"Date {day.date()} is outside dataset range. "
            f"Available interval: {min_date.date()} → {max_date.date()}"
        )

    day_df = df[df["hour"].dt.normalize() == day].copy()

    if day_df.empty:
        raise ValueError(f"No data available for {day.date()}")

    stations = day_df[
        ["station_id", "dpcapacity", "latitude", "longitude", "name"]
    ].drop_duplicates()

    current_bikes = {
        row.station_id: int(np.floor(row.dpcapacity * 0.5))
        for row in stations.itertuples()
    }

    hours = sorted(day_df["hour"].unique())
    results = []

    for hour in hours:

        hour_df = day_df[day_df["hour"] == hour]

        for row in hour_df.itertuples():

            station = row.station_id
            capacity = row.dpcapacity
            demand = row.true_net_demand

            bikes_before = current_bikes[station]

            # raw state before enforcing constraints
            raw_after = bikes_before - demand

            shortage = 0

            if raw_after < 0:
                shortage = abs(raw_after)      # bikes missing
            elif raw_after > capacity:
                shortage = raw_after - capacity  # docks missing

            bikes_after = max(0, min(capacity, raw_after))

            current_bikes[station] = bikes_after

            failure = int((bikes_after == 0) or (bikes_after == capacity))

            results.append(
                {
                    "station_id": station,
                    "hour": hour,
                    "dpcapacity": capacity,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "name": row.name,
                    "current_bike_number": bikes_after,
                    "failure": failure,
                    "failure_magnitude": shortage,
                }
            )

    result_df = pd.DataFrame(results)

    result_df.to_csv(f'..\simulation_results\without_rebalancing\hourly_states_{str(day)[:-8]}.csv', index=False)

    print(f"Simulation saved to simulation_results\without_rebalancing\hourly_states_{str(day)[-8]}.csv")

    return result_df


############################################################################

def simulate_day_with_rebalancing(
    pred_df,
    stations,
    day,
    gamma=10,
    nb_trucks=1,
    acceptable_margin=0.2,
    stations_per_cluster=25,
    opt_time_limit=60
):

    day = pd.to_datetime(day).normalize()

    if not pd.api.types.is_datetime64_any_dtype(pred_df["hour"]):
        raise ValueError("Column 'hour' must be datetime.")

    min_date = pred_df["hour"].min().normalize()
    max_date = pred_df["hour"].max().normalize()

    if not (min_date <= day <= max_date):
        raise ValueError(
            f"Date {day.date()} outside dataset range "
            f"{min_date.date()} → {max_date.date()}"
        )

    day_df = pred_df[pred_df["hour"].dt.normalize() == day].copy()
    hours = sorted(day_df["hour"].unique())

    # initialize bikes at 50%
    current_bikes = {
        row.station_id: int(np.floor(row.dpcapacity * 0.5))
        for row in day_df[["station_id","dpcapacity"]]
        .drop_duplicates()
        .itertuples()
    }

    results = []
    hourly_metrics = []
    optimization_history = {}

    print("Starting simulation with rebalancing")
    print("------------------------------------")

    for hour in hours:

        print(f"\nSimulating hour {hour}")

        h_df = day_df[day_df["hour"] == hour].copy()
        h_df["current_bike_nb"] = h_df["station_id"].map(current_bikes)

        candidate_stations = cluster_candidate_station(
            pred_df=h_df,
            stations=stations,
            hour_to_optimize=hour,
            acceptable_margin=acceptable_margin,
            stations_per_cluster=stations_per_cluster,
            random_init_state=False
        )

        total_distance = 0
        total_P = 0
        total_Q = 0
        trucks_used = 0

        if candidate_stations is None or len(candidate_stations) == 0:

            print("No candidate stations → skipping rebalancing this hour")
            solutions_dict = {}
            optimization_history[hour] = solutions_dict

        else:

            solutions_dict = {}
            nb_clusters = candidate_stations["cluster"].max() + 1
            trucks_used = nb_clusters

            print(f"Clusters to optimize: {nb_clusters}")

            for cluster in range(nb_clusters):

                print(f"Optimizing cluster {cluster}")

                start_time = time.time()

                solution = bike_rebalancing(
                    stations=stations,
                    pred_df=h_df,
                    hour_to_optimize=hour,
                    candidate_stations=candidate_stations,
                    cluster_nb=cluster,
                    gamma=gamma,
                    nb_trucks=nb_trucks,
                    random_init_state=False,
                    opt_time_limit=opt_time_limit
                )

                solutions_dict[cluster] = solution

                duration = round(time.time() - start_time)
                print(f"Cluster {cluster} optimized in {duration}s")

                # --- collect metrics
                W = solution["W"]
                dij = solution["D"]
                P = solution["P"]
                Q = solution["Q"]

                total_distance += sum(
                    value * dij[i][j]
                    for (i,j,k), value in W.items()
                )

                total_P += sum(P.values())
                total_Q += sum(Q.values())

            optimization_history[hour] = solutions_dict

            # ---- apply rebalancing actions
            for sol in solutions_dict.values():

                X = sol["X"]
                Y = sol["Y"]

                for (station_id, truck), value in X.items():
                    current_bikes[station_id] += int(value)

                for (station_id, truck), value in Y.items():
                    current_bikes[station_id] -= int(value)

            # enforce capacity constraints
            for sid in current_bikes:

                capacity = stations.loc[
                    stations["id"] == sid,"dpcapacity"
                ].values[0]

                current_bikes[sid] = max(
                    0, min(capacity, current_bikes[sid])
                )

        # ---- apply real demand
        for row in h_df.itertuples():

            sid = row.station_id
            #demand = row.true_net_demand
            demand = row.pred_net_demand
            capacity = row.dpcapacity

            bikes_before = current_bikes[sid]

            raw_after = bikes_before - demand

            shortage = 0
            if raw_after < 0:
                shortage = abs(raw_after)
            elif raw_after > capacity:
                shortage = raw_after - capacity

            bikes_after = max(0, min(capacity, raw_after))

            current_bikes[sid] = bikes_after

            failure = int(
                (bikes_after == 0) or (bikes_after == capacity)
            )

            results.append({
                "station_id": sid,
                "hour": hour,
                "dpcapacity": capacity,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "name": row.name,
                "current_bike_number": bikes_after,
                "failure": failure,
                "failure_magnitude": shortage
            })

        hourly_metrics.append({
            "hour": hour,
            "total_distance": total_distance,
            "nb_trucks": trucks_used,
            "total_pickups_P": total_P,
            "total_dropoffs_Q": total_Q
        })

        print(f"Hour {hour} completed")

    print("\nSimulation finished")

    results_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(hourly_metrics)

    results_df.to_csv(f'..\simulation_results\with_rebalancing\hourly_states_{str(day)[:-8]}.csv', index=False)

    print(f"Simulation saved to simulation_results\with_rebalancing\hourly_states_{str(day)[-8]}.csv")

    metrics_df.to_csv(f'..\simulation_results\\rebalancing_metrics\hourly_rebalancing_metrics_{str(day)[:-8]}.csv', index=False)

    return results_df, metrics_df, optimization_history