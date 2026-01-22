import os 
import helper
import folium
import pickle
# import cartopy
import numpy as np
# import xarray as xr
import igraph as ig

# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from global_land_mask import globe

from mpl_toolkits.basemap import Basemap

from pyproj import Geod


# WGS84 ellipsoid for geodesic calculations
geod = Geod(ellps="WGS84")
import math
import heapq

from math import pi


def get_ocean_current_dataset():
    """
    Gets the Ocean Currents data: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_1deg

    Returns
    -------
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]

    """
    # url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/oscar/L4/oscar_1_deg/world_oscar_vel_5d2022.nc.gz'
    # try:
    #     ds = netCDF4.Dataset(url)
    # except Exception as e:
    #     print("Received Error while trying to retrieve ocean currents data. \n%s" % (e))
    #     raise e
    # return ds

    with open(os.path.join(os.path.dirname(__file__), "./dataset/ocean-current-dataset-2022.pkl"), 'rb') as f_obj:
        lon, lat, U, V = pickle.load(f_obj)
    
    return lon, lat, U, V
    
def process_ds(lon, lat, U, V):
    """
    Gets the Ocean Currents data: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_1deg

    Parameters
    ----------
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]

    Returns
    -------
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]

    """
    U = U[0, 0, :, :]
    V = V[0, 0, :, :]
    lon[lon>180] = lon[lon>180] - 360
    
    U[np.isnan(U)] = 0.0
    V[np.isnan(V)] = 0.0

    return lon, lat, U, V

def graph_factory(lon, lat, U, V, boat_avg_speed):
    """
    Generates graph with nodes as (lat, lon) 'ocean' points with edge weights as time
    taken by the vessel to traverse it considering the ocean currents

    Parameters
    ----------
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]
    boat_avg_speed: float
        The average speed of the Vessel

    Returns
    -------
    G: igraph.Graph
        The generated graph

    """
    path = os.path.join(os.path.dirname(__file__), "./Graphs/ocean_grid.pkl")
    G_ocean_grid = ig.Graph.Read_Pickle(path)

    boat_avg_speed = float(boat_avg_speed)
    expected_path = os.path.join(os.path.dirname(__file__), "./edge-weight/speed-%s.pkl" % (boat_avg_speed))

    if os.path.exists(expected_path):
        with open(expected_path, 'rb') as weight_obj:
            try:
                edge_weights = pickle.load(weight_obj)
            except EOFError as e: 
                weight_obj.close()
                os.remove(expected_path)
                raise EOFError("The cached edge weight file seems to be corrupted, please try again.")

        G_ocean_grid.es["weight"] = edge_weights

        return G_ocean_grid
    else:
        edge_weights = get_weights(G_ocean_grid, lon, lat, U, V, boat_avg_speed)
        G_ocean_grid.es["weight"] = edge_weights

        return G_ocean_grid

def get_weights(G, lon, lat, U, V, boat_avg_speed):
    """
    Get edge weights to the graph. Weight corresponds to the time
    taken by the vessel to traverse the distance considering the ocean currents.

    Parameters
    ----------
    G: igraph.Graph
        The generated graph
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]
    boat_avg_speed: float
        The average speed of the Vessel

    Returns
    -------
    weights: array
        Array containing the edge weights
    """

    X, Y = np.meshgrid(lon, lat)
    weights = []

    for e in G.es:
        ind_1, ind_2 = e.tuple
        u = helper.get_coord(ind_1, len(lat))
        v = helper.get_coord(ind_2, len(lat))
        weights.append(helper.calculate_cost(X, Y, U, V, (u[1], u[0]), (v[1], v[0]), boat_avg_speed))
    
    return weights    

def get_optimal_routes(graph_object, start_coord, end_coord, lon, lat):
    """
    Generates the shortest path for the ship. (Dijkstra's Algorithm)

    Parameters
    ----------
    graph_object: igraph.Graph
        The generated graph
    start_coord: array
        (lat, lon) of the start point
    end_coord: array
        (lat, lon) of the end point
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points  

    Returns
    -------
    path: array
        List of vertex indices in the optimal path
    """
    start = helper.get_index_from_lat_long(lon, lat, start_coord)
    end = helper.get_index_from_lat_long(lon, lat, end_coord)

    path = graph_object.get_shortest_paths(
            helper.get_node_index(start[1], start[0], len(lat)),
            to=helper.get_node_index(end[1], end[0], len(lat)),
            weights=graph_object.es["weight"],
            output="vpath",
        )
    return np.array([helper.get_coord(res, len(lat)) for res in path[0]])

def get_coordinates_from_path_indices(path, lon, lat):
    """
    Converts the Vertex-indices in path array to array of lat and lon coordinates

    Parameters
    ----------
    path: array
        List of vertex indices in the optimal path
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points  

    Returns
    -------
    xx: array
        1D array of longitude points of the coordinates in the optimal path
    yy: array
        1D array of latitude points of the coordinates in the optimal path
    
    """
    path = path.T
    path[0] = lon[path[0]]
    path[1] = lat[path[1]]
    path = path.astype(np.float64)

    delta = np.diff(path, axis=1)
    mask = (np.abs(delta) > 1)
    row_indices = np.unique(np.nonzero(mask)[1])
    row_indices += 1
    xx, yy = path
    # Adding np.nan incase wrapping around Earth (for plotting)
    xx = np.insert(xx, row_indices, np.nan)
    yy = np.insert(yy, row_indices, np.nan)

    return xx, yy

def sanitize(lon, lat, U, V):
    """
    Sorts longitude and latitude array (Required for plotting)
    """
    
    lon = np.array(lon)
    lat = np.array(lat)
    U = np.array(U)
    V = np.array(V)

    lon_ind = np.argsort(lon)
    lon = lon[lon_ind]
    U = U[:,lon_ind]
    V = V[:,lon_ind]

    lat_ind = np.argsort(lat)
    lat = lat[lat_ind]
    U = U[lat_ind]
    V = V[lat_ind]

    return lon, lat, U, V

def plot_matplot(lon, lat, U, V, xx, yy):
    """
    Generates result plot

    Parameters
    ----------
    lon: array
        1D array containing longitude points
    lat: array
        1D array containing latitude points
    U: array
        2D array containing x component of ocean current speeds [shape -> (len(lat), len(lon))]
    V: array
        2D array containing y component of ocean current speeds [shape -> (len(lat), len(lon))]
    """

    fig, ax = plt.subplots(figsize=(15, 15))
    m = Basemap(width=12000000,height=9000000,resolution='l')
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='aqua', linewidth=0.5)
    m.fillcontinents(color='coral',lake_color='aqua')

    dec = 5
    lon = lon[::dec]
    lat = lat[::dec]
    U = U[::dec, ::dec]
    V = V[::dec, ::dec]
    lon, lat, U, V = sanitize(lon, lat, U, V)

    m.streamplot(lon, lat, U, V, latlon=True, color=U, linewidth=0.5, cmap='ocean', arrowsize=0.5)

    m.plot(xx, yy, 'k:', linewidth=2, label='Optimal Path', latlon=True)
    m.scatter([xx[0]], [yy[0]], c='g', label='Start', latlon=True)
    m.scatter([xx[-1]], [yy[-1]], c='b', label='End', latlon=True)

    plt.legend()
    return fig


def plot_comparison(lon_post, lat_post, U_post, V_post,
                    time_opt_lons, time_opt_lats,
                    start_coord, end_coord):
    """
    Plot both time-optimal and distance-optimal routes on the same map
    """
    # Calculate distance-optimal route
    dist_opt_lons, dist_opt_lats, dist_distance = calculate_distance_optimal_route(
        start_coord, end_coord
    )
   
    # Calculate time-optimal route distance
    time_distance = 0
    for i in range(len(time_opt_lats) - 1):
        time_distance += helper.distance(
            time_opt_lats[i], time_opt_lons[i],
            time_opt_lats[i+1], time_opt_lons[i+1]
        )
   
    # Create figure WITHOUT streamplot to avoid the error
    fig, ax = plt.subplots(figsize=(15, 12))
    m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60,
                llcrnrlon=100, urcrnrlon=220, resolution='l')
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue', linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawparallels(np.arange(-60, 61, 20), labels=[1,0,0,0])
    m.drawmeridians(np.arange(100, 221, 20), labels=[0,0,0,1])
   
    # Plot time-optimal route (current-assisted)
    m.plot(time_opt_lons, time_opt_lats, 'b-', linewidth=3,
          label=f'Time-Optimal Route ({time_distance/1000:.0f} km)', latlon=True)
   
    # Plot distance-optimal route (great-circle)
    m.plot(dist_opt_lons, dist_opt_lats, 'r--', linewidth=3,
          label=f'Distance-Optimal Route ({dist_distance/1000:.0f} km)', latlon=True)
   
    # Plot start and end points
    m.scatter([start_coord[1]], [start_coord[0]], c='green', s=200,
             label='Start (Hastings)', latlon=True, zorder=5, edgecolors='black', linewidth=2)
    m.scatter([end_coord[1]], [end_coord[0]], c='orange', s=200,
             label='End (Kobe)', latlon=True, zorder=5, edgecolors='black', linewidth=2)
   
    plt.legend(loc='upper left', fontsize=12)
    plt.title('Route Comparison: Time-Optimal vs Distance-Optimal\nHastings, Australia to Kobe, Japan',
              fontsize=14, fontweight='bold')
   
    # Print statistics
    print("=" * 60)
    print("ROUTE COMPARISON STATISTICS")
    print("=" * 60)
    print(f"Distance-Optimal Route (Great Circle): {dist_distance/1000:.2f} km")
    print(f"Time-Optimal Route (Current-Assisted): {time_distance/1000:.2f} km")
    print(f"Difference: {(time_distance - dist_distance)/1000:.2f} km")
    print(f"Time-optimal route is {((time_distance/dist_distance - 1) * 100):.1f}% longer")
    print("=" * 60)
   
    return fig

def calculate_distance_optimal_route(start_coord, end_coord, num_points=100):
    """
    Calculate the distance-optimal route (great-circle path) between two points.


    Parameters
    ----------
    start_coord : tuple
        (latitude, longitude) of start point
    end_coord : tuple
        (latitude, longitude) of end point
    num_points : int
        Number of points (including start and end) along the path.


    Returns
    -------
    route_lons : array
        Longitude points along the great-circle route
    route_lats : array
        Latitude points along the great-circle route
    total_distance : float
        Total distance in meters
    """
    start_lat, start_lon = start_coord
    end_lat, end_lon = end_coord


    # Total geodesic distance and azimuth from start to end
    fwd_az, back_az, total_distance = geod.inv(start_lon, start_lat, end_lon, end_lat)


    # Fractions between 0 and 1 for intermediate points
    fractions = np.linspace(0.0, 1.0, num_points)


    route_lats = []
    route_lons = []


    for frac in fractions:
        # Distance along the path at this fraction
        d = total_distance * frac
        lon_i, lat_i, _ = geod.fwd(start_lon, start_lat, fwd_az, d)
        route_lats.append(lat_i)
        route_lons.append(lon_i)


    return np.array(route_lons), np.array(route_lats), total_distance

def great_circle_route(start_coord, end_coord, n=200):
    start_lat, start_lon = start_coord
    end_lat, end_lon = end_coord
    fwd_az, _, total_dist = geod.inv(start_lon, start_lat, end_lon, end_lat)
    fracs = np.linspace(0.0, 1.0, n)
    lats = []
    lons = []
    for f in fracs:
        d = total_dist * f
        lon_i, lat_i, _ = geod.fwd(start_lon, start_lat, fwd_az, d)
        lats.append(lat_i)
        lons.append(lon_i)
    return np.array(lons), np.array(lats), total_dist


def shortest_path(g, src, target):
    """
    Implementation of Dijkstra's Algorithm
    """
    q = [(0, src, ())]
    visited, dist = set(), {src: 0.0}
    while q:
        cost, v, path = heapq.heappop(q)
        print(v, target)
        print("distance left: ", math.sqrt((v[0] - target[0])*(v[0] - target[0]) + (v[1] - target[1])*(v[1] - target[1])))
        if v not in visited:
            visited.add(v)
            path += (v,)
            if v == target:
                return (cost, path)
            
            for cost2, v2 in g.get(v, ()):
                if v2 in visited:
                    continue
                if cost + cost2 < dist.get(v2, float('inf')):
                    dist[v2] = cost + cost2
                    heapq.heappush(q, (cost + cost2, v2, path)) 
    return (float('inf'), ())

def create_graph(x, y):
    """
    Creates Graph with Ocean (lon, lat) as nodes and edges with neighbouring ocean nodes 
    """
    edges = []
    for i in range(len(x)):
        for j in range(len(y)):
            if globe.is_land(y[j], x[i]):
                continue

            center = get_node_index(i, j, len(y))

            if not globe.is_land(y[j], x[(i - 1 + len(x))%len(x)]):
                top = get_node_index((i-1+len(x)) % len(x), j, len(y))
                edges.append((center, top))

            if not globe.is_land(y[j], x[(i+1)%len(x)]):
                bottom = get_node_index((i+1)%len(x), j, len(y))
                edges.append((center, bottom))
                
            if not globe.is_land(y[(j+1)%len(y)], x[i]):
                right = get_node_index(i, (j+1)%len(y), len(y))
                edges.append((center, right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[i]):
                left = get_node_index(i, (j-1+len(y))%len(y), len(y))
                edges.append((center, left))

            if not globe.is_land(y[(j+1)%len(y)], x[(i-1+len(x)) % len(x)]):
                top_right = get_node_index((i-1+len(x)) % len(x), (j+1)%len(y), len(y))
                edges.append((center, top_right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[(i-1+len(x)) % len(x)]):
                top_left = get_node_index((i-1+len(x)) % len(x), (j-1+len(y))%len(y), len(y))
                edges.append((center, top_left))

            if not globe.is_land(y[(j+1)%len(y)], x[(i+1) % len(x)]):
                bottom_right = get_node_index((i+1) % len(x), (j+1)%len(y), len(y))
                edges.append((center, bottom_right))

            if not globe.is_land(y[(j-1+len(y))%len(y)], x[(i+1) % len(x)]):
                bottom_left = get_node_index((i+1) % len(x), (j-1+len(y))%len(y), len(y))
                edges.append((center, bottom_left))

    G = ig.Graph(len(x) * len(y), edges)  
    return G

def calculate_cost(X, Y, U, V, v1, v2, s0):
    """
    Calculates time taken by vessel to travel a distance considering ocean currents
    """
    j1, i1 = v1
    j2, i2 = v2
    
    u = (U[j1,i1] + U[j2,i2])/2.
    v = (V[j1,i1] + V[j2,i2])/2.
    
    ds = distance(Y[v1], X[v1], Y[v2], X[v2])
    a = bearing(Y[v1], X[v1], Y[v2], X[v2])
    
    # Velocity along track
    s = s0 + u*np.cos(a) + v*np.sin(a)

    if s < 0:
        return np.inf
    else:
        return ds/s

def path_distance_and_bearings(lons, lats):
    """
    Compute total great-circle distance and per-segment bearings
    along a polyline defined by (lons, lats).

    Parameters
    ----------
    lons : array-like
        Longitudes of the path (degrees).
    lats : array-like
        Latitudes of the path (degrees).

    Returns
    -------
    total_dist : float
        Total great-circle distance along the path (meters).
    segment_dists : np.ndarray
        Array of per-segment distances (meters), length N-1.
    segment_bearings : np.ndarray
        Array of per-segment bearings (radians), length N-1.
        Bearing is from point i to point i+1.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    n = len(lats)
    if n < 2:
        return 0.0, np.array([]), np.array([])

    segment_dists = np.zeros(n - 1)
    segment_bearings = np.zeros(n - 1)

    total = 0.0
    for i in range(n - 1):
        d = distance(lats[i], lons[i], lats[i+1], lons[i+1])
        b = bearing(lats[i], lons[i], lats[i+1], lons[i+1])
        segment_dists[i] = d
        segment_bearings[i] = b
        total += d

    return total, segment_dists, segment_bearings

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Great-circle distance
    """
    # http://www.movable-type.co.uk/scripts/latlong.html
    R = 6.371e6
    lat1 *= pi/180.
    lon1 *= pi/180.
    lat2 *= pi/180.
    lon2 *= pi/180.
    return R*np.arccos(
        np.sin(lat1)*np.sin(lat2) + 
        np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculates Bearing (angle)
    """
    lat1 *= pi/180.
    lon1 *= pi/180.
    lat2 *= pi/180.
    lon2 *= pi/180.
    y = np.sin(lon2-lon1)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return (pi/2) - np.arctan2(y, x)

def get_node_index(i, j, len_y):
    """
    2D Index -> 1D Index
    """
    return i * len_y + j

def get_coord(index, len_y):
    """
    1D Index -> 2D Index
    """
    return (index // len_y, index % len_y)

def get_index_from_lat_long(x, y, coord):
    """
    coord: list
        (latitude, longitude)
    """
    return (np.absolute(y - coord[0]).argmin(), np.absolute(x - coord[1]).argmin())

def get_distance(v1, v2):
    """
    Calculate the Great-circle distance
    """
    return (np.linalg.norm(v1-v2))

def great_circle_route(start_coord, end_coord, num_points):
    """
    Generate intermediate points along the great-circle (distance-optimal) route.

    Parameters
    ----------
    start_coord : tuple
        (latitude, longitude) of start point (degrees)
    end_coord : tuple
        (latitude, longitude) of end point (degrees)
    num_points : int
        Number of points (including start and end) along the path.

    Returns
    -------
    route_lons : np.ndarray
        Longitudes of points along the great-circle route (degrees)
    route_lats : np.ndarray
        Latitudes of points along the great-circle route (degrees)
    total_distance : float
        Total great-circle distance in meters
    """
    start_lat, start_lon = start_coord
    end_lat, end_lon = end_coord

    # Total geodesic distance and forward azimuth
    fwd_az, back_az, total_distance = _geod.inv(start_lon, start_lat, end_lon, end_lat)

    # Fractions from 0 to 1
    fractions = np.linspace(0.0, 1.0, num_points)

    lats = []
    lons = []

    for frac in fractions:
        d = total_distance * frac
        lon_i, lat_i, _ = _geod.fwd(start_lon, start_lat, fwd_az, d)
        lats.append(lat_i)
        lons.append(lon_i)

    return np.array(lons), np.array(lats), total_distance

def densify_route(lons, lats, num_points):
    """
    Densify a polyline route (lon, lat arrays) into 'num_points' points,
    interpolated along cumulative great-circle distance.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    # cumulative distances along original route
    cumdist = np.zeros(len(lons))
    for i in range(1, len(lons)):
        cumdist[i] = cumdist[i-1] + distance(lats[i-1], lons[i-1], lats[i], lons[i])

    total_dist = cumdist[-1]
    if total_dist == 0 or len(lons) < 2:
        return lons, lats, total_dist

    # target distances
    target = np.linspace(0.0, total_dist, num_points)

    new_lons = []
    new_lats = []

    for d in target:
        j = np.searchsorted(cumdist, d)
        if j == 0:
            new_lons.append(lons[0])
            new_lats.append(lats[0])
        elif j >= len(cumdist):
            new_lons.append(lons[-1])
            new_lats.append(lats[-1])
        else:
            # fraction along segment [j-1, j]
            frac = (d - cumdist[j-1]) / (cumdist[j] - cumdist[j-1])
            lat1, lon1 = lats[j-1], lons[j-1]
            lat2, lon2 = lats[j],   lons[j]

            # interpolate along great-circle segment
            seg_dist = cumdist[j] - cumdist[j-1]
            bearing_ = bearing(lat1, lon1, lat2, lon2)
            d_seg = seg_dist * frac

            # simple forward step: same R and spherical formula as in distance()
            R = 6.371e6
            lat1r = lat1 * pi/180.0
            lon1r = lon1 * pi/180.0
            brg = bearing_

            lat2r = np.arcsin(np.sin(lat1r)*np.cos(d_seg/R) +
                              np.cos(lat1r)*np.sin(d_seg/R)*np.cos(brg))
            lon2r = lon1r + np.arctan2(np.sin(brg)*np.sin(d_seg/R)*np.cos(lat1r),
                                       np.cos(d_seg/R)-np.sin(lat1r)*np.sin(lat2r))

            new_lats.append(lat2r*180.0/pi)
            new_lons.append(lon2r*180.0/pi)

    return np.array(new_lons), np.array(new_lats), total_dist



