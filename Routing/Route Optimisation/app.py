from flask import Flask, render_template, request, redirect, url_for, send_file
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import folium
import time
import psutil
import plotly.graph_objs as go
import plotly.express as px
import io
import base64
import csv

app = Flask(__name__)

# Load the GeoJSON file
geojson_file = "25.geojson"
gdf = gpd.read_file(geojson_file)

# Global variables to store performance metrics and route data
performance_metrics = {}
route_coordinates = []

# Convert GeoJSON to a NetworkX graph
def geojson_to_graph(gdf):
    G = nx.Graph()
    for feature in gdf.geometry:
        coords = list(feature.coords)
        for i in range(len(coords) - 1):
            G.add_edge(coords[i], coords[i + 1], weight=Point(coords[i]).distance(Point(coords[i + 1])))
    return G

G = geojson_to_graph(gdf)

# Define port coordinates
ports = {
    "Port of Busan": (129.040000, 35.100000),
    "Port of Sydney": (151.2333, -33.9500),
    "Port of Hastings": (145.2167, -38.3000),
    "Port of Kobe": (135.1900, 34.6800),
    "Port of Brisbane": (153.1700, -27.3800),
    "Port of Gladstone": (151.2500, -23.8500),
    "Port of Townsville": (146.8200, -19.2500),
    "Port of Cairns": (145.7800, -16.9200)
}

# Snap coordinates to nearest graph node
def snap_to_nearest_node(graph, coord):
    nearest_node = min(graph.nodes, key=lambda x: Point(x).distance(Point(coord)))
    return nearest_node

# Find the shortest path using Dijkstra
def find_shortest_path_dijkstra(source, destination):
    source_coords = ports[source]
    destination_coords = ports[destination]

    source_nearest = snap_to_nearest_node(G, source_coords)
    destination_nearest = snap_to_nearest_node(G, destination_coords)

    start_time = time.time()
    shortest_path = nx.shortest_path(G, source=source_nearest, target=destination_nearest, weight="weight")
    dijkstra_time = time.time() - start_time
    dijkstra_memory = psutil.Process().memory_info().rss

    return shortest_path, dijkstra_time, dijkstra_memory

# Find the shortest path using A*
def find_shortest_path_astar(source, destination):
    source_coords = ports[source]
    destination_coords = ports[destination]

    source_nearest = snap_to_nearest_node(G, source_coords)
    destination_nearest = snap_to_nearest_node(G, destination_coords)

    def heuristic(u, v):
        lat1, lon1 = u
        lat2, lon2 = v
        return Point((lat1, lon1)).distance(Point((lat2, lon2)))

    start_time = time.time()
    shortest_path = nx.astar_path(G, source=source_nearest, target=destination_nearest, heuristic=heuristic, weight="weight")
    astar_time = time.time() - start_time
    astar_memory = psutil.Process().memory_info().rss

    return shortest_path, astar_time, astar_memory

# Find the shortest path using Bellman-Ford
def find_shortest_path_bellmanford(source, destination):
    source_coords = ports[source]
    destination_coords = ports[destination]

    source_nearest = snap_to_nearest_node(G, source_coords)
    destination_nearest = snap_to_nearest_node(G, destination_coords)

    start_time = time.time()
    shortest_path = nx.single_source_bellman_ford_path(G, source_nearest)[destination_nearest]
    bellmanford_time = time.time() - start_time
    bellmanford_memory = psutil.Process().memory_info().rss

    return shortest_path, bellmanford_time, bellmanford_memory

# NEW: Extract coordinates from path
def extract_route_coordinates(path):
    """
    Extract lat/lon coordinates from the path.
    Returns a list of dicts with point_number, longitude, latitude.
    """
    coordinates = []
    for idx, coord in enumerate(path, start=1):
        coordinates.append({
            'point_number': idx,
            'longitude': coord[0],
            'latitude': coord[1]
        })
    return coordinates

# Plot the route on a map
def plot_route(source, destination, path):
    source_coords = ports[source]
    m = folium.Map(location=source_coords, zoom_start=4)
    folium.Marker([source_coords[1], source_coords[0]], popup=source, icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([ports[destination][1], ports[destination][0]], popup=destination, icon=folium.Icon(color="red")).add_to(m)

    route_coords = [(x[1], x[0]) for x in path]
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)

    return m._repr_html_()

# Function to generate comparative graphs using Plotly
def generate_comparative_graph(dijkstra_time, astar_time, bellmanford_time, dijkstra_memory, astar_memory, bellmanford_memory):
    # Time comparison graph
    time_data = [dijkstra_time, astar_time, bellmanford_time]
    time_labels = ['Dijkstra', 'A*', 'Bellman-Ford']

    time_fig = go.Figure([go.Bar(x=time_labels, y=time_data, marker=dict(color=['green', 'blue', 'red']))])
    time_fig.update_layout(title='Time Complexity Comparison', xaxis_title='Algorithm', yaxis_title='Time (seconds)')

    # Space comparison graph
    memory_data = [dijkstra_memory, astar_memory, bellmanford_memory]
    memory_fig = go.Figure([go.Bar(x=time_labels, y=memory_data, marker=dict(color=['green', 'blue', 'red']))])
    memory_fig.update_layout(title='Space Complexity Comparison', xaxis_title='Algorithm', yaxis_title='Memory Usage (bytes)')

    # Convert to base64 for embedding in HTML
    time_graph = time_fig.to_html(full_html=False)
    memory_graph = memory_fig.to_html(full_html=False)

    return time_graph, memory_graph

@app.route("/", methods=["GET", "POST"])
def index():
    global performance_metrics, route_coordinates  # Access the global variables
    map_html = None
    best_algorithm = None
    time_taken = None
    memory_used = None

    if request.method == "POST":
        source = request.form.get("source")
        destination = request.form.get("destination")
        if source in ports and destination in ports:
            # Measure Dijkstra's performance
            dijkstra_path, dijkstra_time, dijkstra_memory = find_shortest_path_dijkstra(source, destination)
            # Measure A* performance
            astar_path, astar_time, astar_memory = find_shortest_path_astar(source, destination)
            # Measure Bellman-Ford's performance
            bellmanford_path, bellmanford_time, bellmanford_memory = find_shortest_path_bellmanford(source, destination)

            # Store the performance metrics
            performance_metrics = {
                "dijkstra_time": dijkstra_time,
                "astar_time": astar_time,
                "bellmanford_time": bellmanford_time,
                "dijkstra_memory": dijkstra_memory,
                "astar_memory": astar_memory,
                "bellmanford_memory": bellmanford_memory,
            }

            # Determine best path (using fastest algorithm)
            if dijkstra_time <= astar_time and dijkstra_time <= bellmanford_time:
                best_path = dijkstra_path
                best_algorithm = "Best algorithm based on time complexity: Dijkstra's Algorithm."
            elif astar_time <= dijkstra_time and astar_time <= bellmanford_time:
                best_path = astar_path
                best_algorithm = "Best algorithm based on time complexity: A* Algorithm."
            else:
                best_path = bellmanford_path
                best_algorithm = "Best algorithm based on time complexity: Bellman-Ford Algorithm."

            # Extract and store route coordinates
            route_coordinates = extract_route_coordinates(best_path)

            # Plot the route using the best algorithm
            map_html = plot_route(source, destination, best_path)

            # Time and memory usage for each algorithm
            time_taken = f"Dijkstra: {dijkstra_time:.6f}s, A*: {astar_time:.6f}s, Bellman-Ford: {bellmanford_time:.6f}s"
            memory_used = f"Dijkstra: {dijkstra_memory} bytes, A*: {astar_memory} bytes, Bellman-Ford: {bellmanford_memory} bytes"

        else:
            map_html = "<p>Invalid port names selected. Please try again.</p>"

    return render_template("index.html", ports=ports.keys(), map_html=map_html, best_algorithm=best_algorithm, time_taken=time_taken, memory_used=memory_used)

@app.route("/analysis", methods=["GET"])
def analysis():
    if not performance_metrics:
        return "<p>No performance data available. Please calculate performance metrics on the main page first.</p>"

    # Use the stored performance metrics
    dijkstra_time = performance_metrics["dijkstra_time"]
    astar_time = performance_metrics["astar_time"]
    bellmanford_time = performance_metrics["bellmanford_time"]
    dijkstra_memory = performance_metrics["dijkstra_memory"]
    astar_memory = performance_metrics["astar_memory"]
    bellmanford_memory = performance_metrics["bellmanford_memory"]

    # Generate comparison graphs
    time_graph, memory_graph = generate_comparative_graph(
        dijkstra_time, astar_time, bellmanford_time,
        dijkstra_memory, astar_memory, bellmanford_memory
    )

    return render_template("analysis.html", time_graph=time_graph, memory_graph=memory_graph)

# NEW: Download route coordinates as CSV
@app.route("/download_route", methods=["GET"])
def download_route():
    global route_coordinates
    
    if not route_coordinates:
        return "<p>No route data available. Please calculate a route first.</p>"
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['point_number', 'longitude', 'latitude'])
    writer.writeheader()
    writer.writerows(route_coordinates)
    
    # Convert to bytes for download
    output.seek(0)
    byte_output = io.BytesIO()
    byte_output.write(output.getvalue().encode('utf-8'))
    byte_output.seek(0)
    
    return send_file(
        byte_output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='ship_route_coordinates.csv'
    )

if __name__ == "__main__":
    app.run(debug=True)
