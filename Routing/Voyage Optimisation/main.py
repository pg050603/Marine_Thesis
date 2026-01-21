import os 
import helper
import folium
import pickle
# import cartopy
import numpy as np
# import xarray as xr
import igraph as ig
import streamlit as st
# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from global_land_mask import globe
from streamlit_folium import st_folium
from mpl_toolkits.basemap import Basemap

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
    Generates result plot focused on the route region.
    """

    all_lons = np.concatenate([lon, np.asarray(xx)])
    all_lats = np.concatenate([lat, np.asarray(yy)])

    margin = 5
    llcrnrlon = float(all_lons.min() - margin)
    urcrnrlon = float(all_lons.max() + margin)
    llcrnrlat = float(all_lats.min() - margin)
    urcrnrlat = float(all_lats.max() + margin)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    m = Basemap(projection="cyl",
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                resolution="l", ax=ax)

    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color="aqua", linewidth=0.5)
    m.fillcontinents(color="coral", lake_color="aqua")

    dec = 5
    lon_dec = lon[::dec]
    lat_dec = lat[::dec]
    U_dec = U[::dec, ::dec]
    V_dec = V[::dec, ::dec]
    lon_dec, lat_dec, U_dec, V_dec = sanitize(lon_dec, lat_dec, U_dec, V_dec)

    m.streamplot(lon_dec, lat_dec, U_dec, V_dec,
                 latlon=True, color=U_dec, linewidth=0.5,
                 cmap="ocean", arrowsize=0.5)

    # downsample route for plotting so dashes are not crowded
    step = max(len(xx) // 200, 1)
    xx_plot = xx[::step]
    yy_plot = yy[::step]

    m.plot(xx_plot, yy_plot,
           color="k", linestyle="-", linewidth=2,
           label="Optimal Path", latlon=True)

    m.scatter([xx[0]], [yy[0]], c="g", label="Start",
              latlon=True, zorder=5)
    m.scatter([xx[-1]], [yy[-1]], c="b", label="End",
              latlon=True, zorder=5)

    plt.legend(loc="upper left")
    return fig



def st_sidebar():
    boat_avg_speed = st.sidebar.number_input('Vessel Speed (m/s)', min_value=0.1, max_value=150.0, step=5.0, value=1.0)

    s_col1, s_col2 = st.sidebar.columns(2)
    with s_col1:
        s_lon = st.number_input("Start Lon", value=89.45)
    with s_col2:
        s_lat = st.number_input("Start Lat", value=14.37)

    with s_col1:
        e_lon = st.number_input("End Lon", value=-32.18)
    with s_col2:
        e_lat = st.number_input("End Lat", value=5.37)
    
    with st.sidebar.expander("Lat-Lon Finder"):
        m = folium.Map(min_lot=-180,
                max_lot=180,
                max_bounds=True,
                min_zoom = 0, zoom_start = 1)
        folium.LatLngPopup().add_to(m)
        folium.Marker((s_lat, s_lon), tooltip='Start Location').add_to(m)
        folium.Marker((e_lat, e_lon), tooltip='End Location').add_to(m)
        st_data = st_folium(m, width="350", height="150")

    return (s_lat, s_lon), (e_lat, e_lon), boat_avg_speed

def st_ui():
    st.write("#Ship-Route Optimization")
    st.markdown(
        """
        When traveling on the surface of the Earth one cannot take a constant heading (an angle with respect to North) to travel the shortest route from point __A__ to __B__. \n
        Instead, the heading must be constantly readjusted so that the arc of the trajectory corresponds to the intersection between the globe and a plane that passes through the center of the Earth. \n
        For ships the ocean currents are an important factor.
        """
    )

    start_coord, end_coord, boat_avg_speed = st_sidebar()
    placeholder = st.empty()

    if (not globe.is_land(start_coord[0], start_coord[1])) and (not globe.is_land(end_coord[0], end_coord[1])):
        generate_btn = placeholder.button("Generate Best Path", disabled=False)
    else:
        st.sidebar.write("Invalid Ocean Coordinates!")
        generate_btn = placeholder.button("Generate Best Path", disabled=True)

    if generate_btn:
        with st.spinner("Fetching Ocean Current Dataset..."):
            # ds = get_ocean_current_dataset()
            # lon, lat, U, V = process_ds(ds)
            lon, lat, U, V = get_ocean_current_dataset()
            lon, lat, U, V = process_ds(lon, lat, U, V)

        with st.spinner("Creating the latitude-longitude Graph..."):
            G = graph_factory(lon, lat, U, V, boat_avg_speed)

        with st.spinner("Generating Shortest Path..."):
            path = get_optimal_routes(G, start_coord, end_coord, lon, lat)
            xx, yy = get_coordinates_from_path_indices(path, lon, lat)
            
        with st.spinner("Plotting results..."):
            fig = plot_matplot(lon, lat, U, V, xx, yy)
            st.pyplot(fig)
            # fig_html = mpld3.fig_to_html(fig)
            # components.html(fig_html, height=600)


if __name__ == '__main__':
    st_ui()

