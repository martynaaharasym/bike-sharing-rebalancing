import folium
import pandas as pd
import numpy as np

def visualize_cluster(candidate_stations,hour_to_optimize):
    center_lat = candidate_stations['latitude'].mean()
    center_lon = candidate_stations['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'yellow', 'pink', 'cyan', 'magenta', 'lime', 
        'teal', 'brown', 'navy', 'gold', 'violet', 
        'indigo', 'turquoise', 'coral', 'salmon', 'olive'
    ]

    for _, row in candidate_stations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=colors[row['cluster']],
            fill=True,
            fill_color=colors[row['cluster']],
            fill_opacity=0.8,
            tooltip=f"Station {row['id']} | Cluster {row['cluster']}"
        ).add_to(m)

    m.save(f"..\plots\clusters_map\station_clusters_{hour_to_optimize[:-6]}.html")
    print(f'station_clusters_{hour_to_optimize[:-6]}.html saved in plots/clusters_map')



def print_truck_routes_report(solutions_dict):
    
    print("\n===== TRUCK ROUTE REPORT (ALL CLUSTERS) =====\n")

    global_truck_id = 0

    for cluster, sol in solutions_dict.items():
        
        I = sol["I"]
        I_all = sol["I_all"]
        K = sol["K"]
        warehouse_id = sol["warehouse_id"]
        D = sol["D"]
        
        X = sol["X"]
        Y = sol["Y"]
        T = sol["T"]
        W = sol["W"]
        
        for k in K:
            
            print(f"\nTruck {global_truck_id} (cluster {cluster}) :")
            
            current = warehouse_id
            visited = set()
            total_km = 0
            
            while True:
                
                next_station = None
                
                for j in I_all:
                    if W[current, j, k] > 0.5:
                        next_station = j
                        break
                
                if next_station is None:
                    print("  No route.")
                    break
                
                total_km += D[current][next_station]
                
                if next_station == warehouse_id:
                    print(f"  Goes from {current} -> Warehouse (return) | "
                        f"Distance: {D[current][next_station]:.2f} km")
                    break
                
                drop = X.get((next_station, k), 0)
                pick = Y.get((next_station, k), 0)
                load = T[(next_station, k)]
                
                print(f"  Goes from {current} -> {next_station} | "
                    f"Distance: {D[current][next_station]:.2f} km | "
                    f"Drop: {drop:.0f}, Pick: {pick:.0f}, "
                    f"Truck load before arrival: {load:.0f}")
                
                if next_station in visited:
                    print("  WARNING: loop detected")
                    break
                
                visited.add(next_station)
                current = next_station
            
            print(f"  --> Total distance traveled by Truck {global_truck_id}: {total_km:.2f} km")
            
            global_truck_id += 1


def visualize_truck_routes(stations, solutions_dict, hour_to_optimize) : 
    center_lat = stations['latitude'].mean()
    center_lon = stations['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    truck_colors = [
        'blue','red','green','orange','purple',
        'cyan','magenta','brown','pink','black'
    ]

    global_truck_id = 0
    offset_step = 0.00005

    for cluster, sol in solutions_dict.items():
        
        I = sol["I"]
        I_all = sol["I_all"]
        K = sol["K"]
        warehouse_id = sol["warehouse_id"]
        
        X = sol["X"]
        Y = sol["Y"]
        W = sol["W"]
        
        for k in K:
            
            color = truck_colors[global_truck_id % len(truck_colors)]
            
            # --- reconstruct route ---
            route = [warehouse_id]
            visited = set()
            current = warehouse_id
            
            while True:
                
                next_stations = [j for j in I_all if W[current, j, k] > 0.5]
                
                if len(next_stations) == 0:
                    break
                
                next_station = next_stations[0]
                
                if next_station == warehouse_id:
                    route.append(warehouse_id)
                    break
                
                if next_station in visited:
                    break
                
                route.append(next_station)
                visited.add(next_station)
                current = next_station
            
            # --- draw route ---
            coords = [
                (
                    stations.loc[stations['id']==s, 'latitude'].values[0],
                    stations.loc[stations['id']==s, 'longitude'].values[0]
                )
                for s in route
            ]
            
            folium.PolyLine(
                coords,
                color=color,
                weight=4,
                opacity=0.7,
                tooltip=f"Truck {global_truck_id} (cluster {cluster})"
            ).add_to(m)
            
            # --- draw station actions ---
            for s in route:
                
                if s == warehouse_id:
                    continue
                
                lat = stations.loc[stations['id']==s,'latitude'].values[0]
                lon = stations.loc[stations['id']==s,'longitude'].values[0]
                
                lat_offset = lat + global_truck_id * offset_step
                lon_offset = lon + global_truck_id * offset_step
                
                drop = X.get((s,k),0)
                pick = Y.get((s,k),0)
                
                tooltip = f"Truck {global_truck_id}: "
                
                if drop > 0:
                    tooltip += f"Drop {drop:.0f}"
                    
                    folium.CircleMarker(
                        location=[lat_offset, lon_offset],
                        radius=5,
                        color='green',
                        fill=True,
                        fill_color='green',
                        fill_opacity=0.7,
                        tooltip=tooltip
                    ).add_to(m)
                
                if pick > 0:
                    tooltip += f" Pick {pick:.0f}"
                    
                    folium.CircleMarker(
                        location=[lat_offset, lon_offset],
                        radius=5,
                        color='orange',
                        fill=True,
                        fill_color='orange',
                        fill_opacity=0.7,
                        tooltip=tooltip
                    ).add_to(m)
            
            global_truck_id += 1

    # --- Save and open map ---
    m.save(f"..\plots\\truck_routes_map\\truck_routes_map_{hour_to_optimize[:-6]}.html")
    print(f'truck_routes_map_{hour_to_optimize[:-6]}.html saved in plots/truck_routes_map')