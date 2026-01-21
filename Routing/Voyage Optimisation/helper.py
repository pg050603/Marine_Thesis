import math
import heapq
import numpy as np
from math import pi
import igraph as ig
from global_land_mask import globe
from pyproj import Geod  # add this


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



# WGS84 ellipsoid for geodesic calculations
_geod = Geod(ellps="WGS84")


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
