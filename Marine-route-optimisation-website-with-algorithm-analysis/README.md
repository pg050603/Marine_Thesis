# Maritime Route Optimization Web Application

This project is a web-based application for optimizing maritime transport routes in the Indian subcontinent. It enables users to compare the performance of three shortest path algorithms—Dijkstra’s, A*, and Bellman-Ford—using real-world geospatial and oceanographic data. The application visualizes optimal routes between ports and provides interactive analysis tools.

## Features

- Interactive maritime map with port-to-port routing
- Implements and compares three shortest path algorithms:
  - Dijkstra’s Algorithm
  - A* Algorithm (heuristic-based)
  - Bellman-Ford Algorithm (supports negative weights)
- Computes and compares time and memory usage for each algorithm
- Visualizes routes and performance metrics with dynamic graphs and maps
- Integrates real-world oceanographic data from CMEMS (Copernicus Marine Environment Monitoring Service)

## Technology Stack

**Backend:**
- Python (Flask)
- NetworkX: Graph representation and pathfinding
- GeoPandas, Shapely: GeoJSON and spatial data processing
- Plotly: Data visualization
- psutil: Memory tracking
- xarray, ftplib: Oceanographic data handling

**Frontend:**
- HTML templates rendered by Flask
- Folium for interactive maps
- CSS for styling

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/maritime-route-optimizer.git
   cd maritime-route-optimizer
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Open your browser:**
   Go to [http://localhost:5000](http://localhost:5000)

## Application Workflow

1. The user selects source and destination ports through the web interface.
2. GeoJSON and oceanographic data are processed to build a weighted graph.
3. The selected algorithms compute optimal paths based on distance, time, and memory.
4. The application displays the route on an interactive map and provides comparison charts for performance evaluation.

## Evaluation Summary

- **A\*** algorithm performed fastest in terms of execution time.
- **Dijkstra’s** algorithm was the most efficient in terms of memory usage.
- **Bellman-Ford** algorithm was the most flexible for handling graphs with negative weights but was less efficient overall.

## Use Cases

- Maritime route planning and decision-making
- Evaluation of algorithm efficiency for transportation systems
- Support for logistics and shipping industries
- Academic demonstration of graph-based optimization

## Data Source

- **Copernicus Marine Environment Monitoring Service (CMEMS)**
- Key variables:
  - Significant wave height
  - Wind wave height
  - Mixed layer thickness
  - Salinity
  - Temperature

## File Structure

- `app.py` – Main Flask application
- `static/` – Static assets (CSS, images)
- `templates/` – HTML templates
- `25.geojson` – Geospatial data for ports/routes

