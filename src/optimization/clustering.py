import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained

def cluster_candidate_station(pred_df, stations, hour_to_optimize, acceptable_margin:float=0.2, stations_per_cluster:int=25, random_init_state=False):

    # filtering candidate stations
    hto = pred_df[pred_df['hour']==hour_to_optimize]
    target = 0.5 * hto['dpcapacity']

    if random_init_state == True : 
        np.random.seed(8)
        hto['current_bike_nb'] = (np.clip(
                np.random.normal(0.5 * hto['dpcapacity'], 0.2 * hto['dpcapacity']),
                0,
                hto['dpcapacity']
            )).astype(int)
    
    deviation = (hto['current_bike_nb'] - target).abs()
    candidate_stations_ids = hto.loc[
        deviation >= acceptable_margin * hto['dpcapacity'],
        'station_id'
    ].tolist()
    candidate_stations = stations[stations['id'].isin(candidate_stations_ids)]

    if (candidate_stations is not None) and len(candidate_stations) != 0:
        
        # clustering
        n_clusters = max(1,int(np.floor(len(candidate_stations)/stations_per_cluster)))
        if n_clusters > 1 : 
            size_min = stations_per_cluster - 5
            if size_min * n_clusters > len(candidate_stations):
                size_min = int(np.floor(len(candidate_stations) / n_clusters))
            size_max = stations_per_cluster + 5
            if size_max * n_clusters < len(candidate_stations):
                size_max = int(np.ceil(len(candidate_stations)/n_clusters))
        else : 
            size_min = len(candidate_stations)
            size_max = len(candidate_stations)

        coords = candidate_stations[['latitude', 'longitude']]

        kmeans_constrained = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=8, n_init=20)
        candidate_stations['cluster'] = kmeans_constrained.fit_predict(coords)

    return candidate_stations