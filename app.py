from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
import folium
import tempfile
import random
import numpy as np
import networkx as nx
from datetime import datetime
from route_optimizer import (
    get_or_load_road_network,
    get_simplified_road_network,
    create_synthetic_network,
    add_base_speeds_to_graph,
    MLTrafficPredictor,
    RoadCongestionClassifier,
    TravelTimePredictor,
    EnhancedDynamicRouter,
    user_location_selection_feature,
    check_graph_connectivity,
    find_nearest_accessible_node,
    create_enhanced_route_map,
    create_route_focused_map,
    display_route_details
)

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'saved_models'
TRAFFIC_MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_predictor.pkl')
CONGESTION_MODEL_PATH = os.path.join(MODEL_DIR, 'congestion_classifier.pkl')
TRAVEL_TIME_MODEL_PATH = os.path.join(MODEL_DIR, 'travel_time_predictor.pkl')

G = None
edge_ids = None
traffic_predictor = None
congestion_classifier = None
travel_predictor = None
router = None
congestion_risk = None

def load_or_train_model(model_class, filepath, *args):
    """Helper function to load model or train if not exists"""
    if os.path.exists(filepath):
        print(f"Loading {filepath}")
        return model_class.load(filepath, *args)
    else:
        print(f"Training and saving {filepath}")
        model = model_class(*args)
        model.save(filepath)
        return model

def initialize_system():
    """Initialize the route optimization system and cache the results"""
    global G, edge_ids, traffic_predictor, congestion_classifier, travel_predictor, router, congestion_risk
    
    if G is None:
        
        # Get road network with better error handling
        try:
            G = get_or_load_road_network()
        except Exception as e:
            print(f"Error obtaining road network: {str(e)}")
            print("Falling back to synthetic network")
            G = create_synthetic_network()
            
        G = add_base_speeds_to_graph(G)
        edge_ids = list(G.edges())

        os.makedirs(MODEL_DIR, exist_ok=True)
        
        traffic_predictor = load_or_train_model(
            MLTrafficPredictor,
            TRAFFIC_MODEL_PATH,
            edge_ids
        )
        
        congestion_classifier = load_or_train_model(
            RoadCongestionClassifier,
            CONGESTION_MODEL_PATH,
            G
        )
        
        travel_predictor = load_or_train_model(
            TravelTimePredictor,
            TRAVEL_TIME_MODEL_PATH,
            G
        )
        
        # Classify congestion risk
        congestion_risk = congestion_classifier.classify_roads()
        
        # Set up router
        router = EnhancedDynamicRouter(G, edge_ids, travel_predictor)

    return {
        "G": G,
        "edge_ids": edge_ids,
        "traffic_predictor": traffic_predictor,
        "congestion_classifier": congestion_classifier,
        "travel_predictor": travel_predictor,
        "router": router,
        "congestion_risk": congestion_risk
    }


@app.route('/api/status', methods=['GET'])
def status():
    """Health check endpoint"""
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat()})

@app.route('/api/network/stats', methods=['GET'])
def get_network_stats():
    """Get basic statistics about the road network"""
    system = initialize_system()
    G = system["G"]
    
    node_count = len(G.nodes())
    edge_count = len(G.edges())
    
    # Get network bounding box
    min_lat = min(G.nodes[node]['y'] for node in G.nodes())
    max_lat = max(G.nodes[node]['y'] for node in G.nodes())
    min_lon = min(G.nodes[node]['x'] for node in G.nodes())
    max_lon = max(G.nodes[node]['x'] for node in G.nodes())
    
    # Road type statistics
    road_types = {}
    for _, _, data in G.edges(data=True):
        road_type = data.get('highway', 'unknown')
        if road_type in road_types:
            road_types[road_type] += 1
        else:
            road_types[road_type] = 1
    
    return jsonify({
        "node_count": node_count,
        "edge_count": edge_count,
        "bounding_box": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        },
        "road_types": road_types
    })
    
    
    
# Add these routes to the Flask backend

@app.route('/api/map/initial', methods=['GET'])
def get_initial_map():
    """Return an initial map centered on the network"""
    system = initialize_system()
    G = system["G"]
    
    # Calculate network center 
    lat_values = [data['y'] for _, data in G.nodes(data=True) if 'y' in data]
    lon_values = [data['x'] for _, data in G.nodes(data=True) if 'x' in data]
    
    if not lat_values or not lon_values:
        # Default center if no nodes with coordinates'        
        center_lat, center_lon = 0, 0
    else:
        center_lat = sum(lat_values) / len(lat_values)
        center_lon = sum(lon_values) / len(lon_values)
    
    # Create a basic map centered on the network
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add a sample of network nodes for reference (limit to avoid heavy map)
    node_sample = random.sample(list(G.nodes()), min(100, len(G.nodes())))
    
    for node_id in node_sample:
        node_data = G.nodes[node_id]
        if 'y' in node_data and 'x' in node_data:
            folium.CircleMarker(
                location=[node_data['y'], node_data['x']],
                radius=2,
                color='blue',
                fill=True,
                fill_opacity=0.7,
                popup=f"Node ID: {node_id}"
            ).add_to(m)
    
    # Add JavaScript to handle click events on the map
# Add JavaScript to handle click events on the map with visual feedback
    click_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = document.querySelector('.folium-map');
        
        // Ensure map exists and has been initialized
        if (!map || !map._leaflet_id) {
            console.error('Map element not found or not initialized');
            return;
        }
        
        // Access the Leaflet map instance
        var leafletMap = window.L.DomUtil.get('map')._leaflet_map || map._leaflet;
        
        // Add click event listener to the map with clear visual feedback
        leafletMap.on('click', function(e) {
            // Get lat/lon from the click event
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;
            
            // Add a marker to show the selected point
            var marker = L.marker([lat, lon], {
                icon: L.divIcon({
                    className: 'selected-node-marker',
                    html: '<div style="background-color: red; width: 10px; height: 10px; border-radius: 50%; border: 2px solid white;"></div>'
                })
            }).addTo(leafletMap);
            
            // Send message to parent window with node selection info
            window.parent.postMessage({
                type: 'NODE_SELECTED',
                data: {
                    lat: lat,
                    lon: lon
                }
            }, '*');
            
            // Show visual feedback about the selection
            var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent('<p>Point selected at ' + lat.toFixed(6) + ', ' + lon.toFixed(6) + '</p>')
                .openOn(leafletMap);
                
            // Optional: Make an API call to get nearby nodes
            fetch('/api/network/nearby-nodes?lat=' + lat + '&lon=' + lon)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.nodes.length > 0) {
                        // Show closest node
                        var closestNode = data.nodes[0];
                        L.circleMarker([closestNode.lat, closestNode.lon], {
                            radius: 8,
                            color: 'blue',
                            fillColor: '#3186cc',
                            fillOpacity: 0.7
                        }).addTo(leafletMap)
                        .bindPopup('Selected Node ID: ' + closestNode.id)
                        .openPopup();
                    }
                })
                .catch(error => console.error('Error fetching nearby nodes:', error));
        });
    });
    </script>
    """


    
    m.get_root().html.add_child(folium.Element(click_script))
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        m.save(tmp.name)
        tmp_filename = tmp.name
    
    return send_file(tmp_filename, mimetype='text/html')

@app.route('/api/map/update-with-nodes', methods=['POST'])
def update_map_with_nodes():
    """Update map to show search results"""
    system = initialize_system()
    G = system["G"]
    
    data = request.json
    nodes = data.get('nodes', [])
    
    if not nodes:
        return jsonify({
            "error": "No nodes provided",
            "success": False
        }), 400
    
    # Calculate map center based on provided nodes
    lat_values = [node['lat'] for node in nodes if 'lat' in node]
    lon_values = [node['lon'] for node in nodes if 'lon' in node]
    
    if not lat_values or not lon_values:
        # Default center if no nodes with coordinates
        lat_values = [data['y'] for _, data in G.nodes(data=True) if 'y' in data]
        lon_values = [data['x'] for _, data in G.nodes(data=True) if 'x' in data]
        
        if not lat_values or not lon_values:
            center_lat, center_lon = 0, 0
        else:
            center_lat = sum(lat_values) / len(lat_values)
            center_lon = sum(lon_values) / len(lon_values)
    else:
        center_lat = sum(lat_values) / len(lat_values)
        center_lon = sum(lon_values) / len(lon_values)
    
    # Create map centered on search results
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Add the search result nodes with distinct styling
    for node in nodes:
        if 'lat' in node and 'lon' in node:
            folium.CircleMarker(
                location=[node['lat'], node['lon']],
                radius=5,
                color='red',
                fill=True,
                fill_opacity=0.7,
                popup=f"Node ID: {node['id']}"
            ).add_to(m)
    
    # Add some surrounding nodes for context
    nearby_nodes = []
    for node_id, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
            for search_node in nodes:
                if 'lat' in search_node and 'lon' in search_node:
                    dist = ((data['y'] - search_node['lat'])**2 + (data['x'] - search_node['lon'])**2)**0.5
                    if dist <= 0.005:  # Small radius in coordinate units
                        nearby_nodes.append((node_id, data))
                        break
    
    # Sample some nearby nodes to avoid cluttering the map
    nearby_sample = random.sample(nearby_nodes, min(30, len(nearby_nodes)))
    
    for node_id, node_data in nearby_sample:
        folium.CircleMarker(
            location=[node_data['y'], node_data['x']],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.3,
            popup=f"Node ID: {node_id}"
        ).add_to(m)
    
    # Add click handler for selecting nodes
  # Add JavaScript to handle click events on the map with visual feedback
    click_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = document.querySelector('.folium-map');
        
        // Ensure map exists and has been initialized
        if (!map || !map._leaflet_id) {
            console.error('Map element not found or not initialized');
            return;
        }
        
        // Access the Leaflet map instance
        var leafletMap = window.L.DomUtil.get('map')._leaflet_map || map._leaflet;
        
        // Add click event listener to the map with clear visual feedback
        leafletMap.on('click', function(e) {
            // Get lat/lon from the click event
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;
            
            // Add a marker to show the selected point
            var marker = L.marker([lat, lon], {
                icon: L.divIcon({
                    className: 'selected-node-marker',
                    html: '<div style="background-color: red; width: 10px; height: 10px; border-radius: 50%; border: 2px solid white;"></div>'
                })
            }).addTo(leafletMap);
            
            // Send message to parent window with node selection info
            window.parent.postMessage({
                type: 'NODE_SELECTED',
                data: {
                    lat: lat,
                    lon: lon
                }
            }, '*');
            
            // Show visual feedback about the selection
            var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent('<p>Point selected at ' + lat.toFixed(6) + ', ' + lon.toFixed(6) + '</p>')
                .openOn(leafletMap);
                
            // Optional: Make an API call to get nearby nodes
            fetch('/api/network/nearby-nodes?lat=' + lat + '&lon=' + lon)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.nodes.length > 0) {
                        // Show closest node
                        var closestNode = data.nodes[0];
                        L.circleMarker([closestNode.lat, closestNode.lon], {
                            radius: 8,
                            color: 'blue',
                            fillColor: '#3186cc',
                            fillOpacity: 0.7
                        }).addTo(leafletMap)
                        .bindPopup('Selected Node ID: ' + closestNode.id)
                        .openPopup();
                    }
                })
                .catch(error => console.error('Error fetching nearby nodes:', error));
        });
    });
    </script>
    """

    
    m.get_root().html.add_child(folium.Element(click_script))
    

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        m.save(tmp.name)
        tmp_filename = tmp.name
    
    return send_file(tmp_filename, mimetype='text/html')

@app.route('/api/map/update-route', methods=['POST'])
def update_route_map():
    """Update map to show selected depot and stops"""
    system = initialize_system()
    G = system["G"]
    congestion_risk = system["congestion_risk"]
    router = system["router"]
    traffic_predictor = system["traffic_predictor"]
    
    data = request.json
    stops = data.get('stops', [])
    depot = data.get('depot')
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 1)
    is_weekend = day_of_week >= 5
    
    if not depot and not stops:
        return jsonify({
            "error": "No depot or stops provided",
            "success": False
        }), 400
    
    all_points = [depot] + stops if depot else stops
    all_points = [p for p in all_points if p is not None]
    
    if len(all_points) < 1:
        return jsonify({
            "error": "Need at least one valid location",
            "success": False
        }), 400
    
    # Check connectivity and find valid nodes if needed
    valid_points = []
    for point in all_points:
        if point in G:
            valid_points.append(point)
        else:
            closest_node = find_nearest_accessible_node(G, point)
            if closest_node:
                valid_points.append(closest_node)
    
    if len(valid_points) < 1:
        return jsonify({
            "error": "No valid nodes found",
            "success": False
        }), 400
    
    # Update traffic for routing
    traffic_multipliers = traffic_predictor.predict_for_time(hour, day_of_week)
    router.update_edge_speeds(traffic_multipliers)
    
    # Generate a route if we have more than one point
    route = None
    if len(valid_points) > 1:
        route = router.optimize_route(valid_points)
    
    # Create map
    if route:
        time_period_name = f"Hour_{hour}_{'Weekend' if is_weekend else 'Weekday'}"
        route_map = create_enhanced_route_map(G, valid_points, route, time_period_name, congestion_risk)
    else:
        # Just create a map showing the points without a route
        lat_values = [G.nodes[p]['y'] for p in valid_points if p in G.nodes()]
        lon_values = [G.nodes[p]['x'] for p in valid_points if p in G.nodes()]
        
        if not lat_values or not lon_values:
            return jsonify({
                "error": "No valid coordinates found for selected nodes",
                "success": False
            }), 400
        
        center_lat = sum(lat_values) / len(lat_values)
        center_lon = sum(lon_values) / len(lon_values)
        
        route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        for i, point_id in enumerate(valid_points):
            if point_id in G.nodes():
                color = 'green' if i == 0 and depot else 'blue'
                folium.CircleMarker(
                    location=[G.nodes[point_id]['y'], G.nodes[point_id]['x']],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{'Depot' if i == 0 and depot else 'Stop'} ID: {point_id}"
                ).add_to(route_map)
    

    click_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var map = document.querySelector('.folium-map');
        
        // Ensure map exists and has been initialized
        if (!map || !map._leaflet_id) {
            console.error('Map element not found or not initialized');
            return;
        }
        
        // Access the Leaflet map instance
        var leafletMap = window.L.DomUtil.get('map')._leaflet_map || map._leaflet;
        
        // Add click event listener to the map with clear visual feedback
        leafletMap.on('click', function(e) {
            // Get lat/lon from the click event
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;
            
            // Add a marker to show the selected point
            var marker = L.marker([lat, lon], {
                icon: L.divIcon({
                    className: 'selected-node-marker',
                    html: '<div style="background-color: red; width: 10px; height: 10px; border-radius: 50%; border: 2px solid white;"></div>'
                })
            }).addTo(leafletMap);
            
            // Send message to parent window with node selection info
            window.parent.postMessage({
                type: 'NODE_SELECTED',
                data: {
                    lat: lat,
                    lon: lon
                }
            }, '*');
            
            // Show visual feedback about the selection
            var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent('<p>Point selected at ' + lat.toFixed(6) + ', ' + lon.toFixed(6) + '</p>')
                .openOn(leafletMap);
                
            // Optional: Make an API call to get nearby nodes
            fetch('/api/network/nearby-nodes?lat=' + lat + '&lon=' + lon)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.nodes.length > 0) {
                        // Show closest node
                        var closestNode = data.nodes[0];
                        L.circleMarker([closestNode.lat, closestNode.lon], {
                            radius: 8,
                            color: 'blue',
                            fillColor: '#3186cc',
                            fillOpacity: 0.7
                        }).addTo(leafletMap)
                        .bindPopup('Selected Node ID: ' + closestNode.id)
                        .openPopup();
                    }
                })
                .catch(error => console.error('Error fetching nearby nodes:', error));
        });
    });
    </script>
    """
    
    route_map.get_root().html.add_child(folium.Element(click_script))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        route_map.save(tmp.name)
        tmp_filename = tmp.name
    
    return send_file(tmp_filename, mimetype='text/html')    
    

@app.route('/api/network/nodes', methods=['GET'])
def get_network_nodes():
    """Get all nodes in the network with their coordinates"""
    system = initialize_system()
    G = system["G"]
    
    limit = int(request.args.get('limit', 100))
    nodes_data = []
    
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        if i >= limit:
            break
            
        nodes_data.append({
            "id": node_id,
            "lat": data.get('y'),
            "lon": data.get('x')
        })
    
    return jsonify({
        "nodes": nodes_data,
        "total": len(G.nodes()),
        "returned": len(nodes_data)
    })

@app.route('/api/traffic/predict', methods=['POST'])
def predict_traffic():
    """Predict traffic conditions based on time"""
    system = initialize_system()
    traffic_predictor = system["traffic_predictor"]
    
    data = request.json
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 1)  # 0=Monday, 6=Sunday
    is_weekend = day_of_week >= 5
    
    # Predict traffic multipliers
    traffic_multipliers = traffic_predictor.predict_for_time(hour, day_of_week)
    
    # Return only a subset of multipliers to avoid large payloads
    return jsonify({
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "traffic_sample": traffic_multipliers[:10].tolist(),  # Convert numpy array to list
        "average_multiplier": float(np.mean(traffic_multipliers)),
        "min_multiplier": float(np.min(traffic_multipliers)),
        "max_multiplier": float(np.max(traffic_multipliers))
    })

@app.route('/api/congestion/classify', methods=['GET'])
def get_congestion():
    """Get road congestion classification"""
    system = initialize_system()
    G = system["G"]
    congestion_risk = system["congestion_risk"]
    
    # Convert edge tuples to strings for JSON serialization
    congestion_data = {}
    for (u, v), risk in congestion_risk.items():
        edge_key = f"{u}-{v}"
        congestion_data[edge_key] = {
            "risk_level": int(risk),
            "risk_label": "Low" if risk == 0 else "Medium" if risk == 1 else "High",
            "from_node": u,
            "to_node": v
        }
    
    # Count by risk level
    risk_counts = {
        "low": sum(1 for r in congestion_risk.values() if r == 0),
        "medium": sum(1 for r in congestion_risk.values() if r == 1),
        "high": sum(1 for r in congestion_risk.values() if r == 2)
    }
    
    # Return only a sample of congestion data to avoid large payloads
    sample_size = min(100, len(congestion_data))
    sample_keys = random.sample(list(congestion_data.keys()), sample_size)
    sample_data = {k: congestion_data[k] for k in sample_keys}
    
    return jsonify({
        "congestion_sample": sample_data,
        "risk_counts": risk_counts,
        "total_roads": len(congestion_risk)
    })

@app.route('/api/route/optimize', methods=['POST'])
def optimize_route():
    """Optimize a delivery route based on stops and time"""
    system = initialize_system()
    G = system["G"]
    router = system["router"]
    traffic_predictor = system["traffic_predictor"]
    congestion_risk = system["congestion_risk"]
    
    data = request.json
    
    # Get time parameters
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 1)
    is_weekend = day_of_week >= 5
    
    # Get stop parameters
    depot_id = data.get('depot_id')
    stop_ids = data.get('stop_ids', [])
    
    # If depot_id is not provided, use first node in the graph
    if depot_id is None:
        depot_id = list(G.nodes())[0]
    
    # If no stops are provided, select random nodes
    if not stop_ids:
        nodes = list(G.nodes())
        if depot_id in nodes:
            nodes.remove(depot_id)
        stop_count = min(5, len(nodes))
        stop_ids = random.sample(nodes, stop_count)
    
    # Combine depot and stops
    all_stops = [depot_id] + stop_ids
    
    # Check connectivity and fix if needed
    if not check_graph_connectivity(G, all_stops):
        new_stops = []
        for stop in all_stops:
            if stop not in G:
                closest_node = find_nearest_accessible_node(G, stop)
                if closest_node:
                    new_stops.append(closest_node)
            else:
                new_stops.append(stop)
        all_stops = new_stops
    
    # Update traffic conditions
    traffic_multipliers = traffic_predictor.predict_for_time(hour, day_of_week)
    router.update_edge_speeds(traffic_multipliers)
    
    # Optimize route
    route = router.optimize_route(all_stops)
    
    if not route:
        return jsonify({
            "error": "Failed to find a valid route",
            "success": False
        }), 400
    
    # Calculate time matrix and details
    time_matrix = router.calculate_time_matrix(all_stops, hour=hour, is_weekend=is_weekend)
    
    # Calculate total time and segment details
    total_time = 0
    segments = []
    
    for i in range(len(route) - 1):
        start_node = route[i]
        end_node = route[i+1]
        
        start_idx = all_stops.index(start_node) if start_node in all_stops else i
        end_idx = all_stops.index(end_node) if end_node in all_stops else i+1
        
        segment_time = time_matrix[start_idx][end_idx]
        
        # Handle unrealistic travel times as shown in display_route_details from second code
        if segment_time > 1000:  # Unrealistic travel time
            start_lat = G.nodes[start_node]['y']
            start_lon = G.nodes[start_node]['x']
            end_lat = G.nodes[end_node]['y']  
            end_lon = G.nodes[end_node]['x']
            
            dist_km = ((start_lat - end_lat)**2 + (start_lon - end_lon)**2)**0.5 * 111
            segment_time = (dist_km / 30) * 60  # Estimate based on distance
            
        total_time += segment_time
        
        segments.append({
            "from_node": start_node,
            "to_node": end_node,
            "time_minutes": float(segment_time),
            "cumulative_time": float(total_time)
        })
    
    # Prepare node details
    nodes_info = []
    for i, node_id in enumerate(route):
        if node_id in G.nodes():
            nodes_info.append({
                "id": node_id,
                "lat": float(G.nodes[node_id].get('y', 0)),
                "lon": float(G.nodes[node_id].get('x', 0)),
                "is_depot": i == 0,
                "stop_number": i
            })
    
    return jsonify({
        "success": True,
        "route": route,
        "total_time_minutes": float(total_time),
        "segments": segments,
        "nodes": nodes_info,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend
    })

@app.route('/api/map/route', methods=['POST'])
def generate_route_map():
    """Generate an HTML map of the optimized route"""
    system = initialize_system()
    G = system["G"]
    router = system["router"]
    traffic_predictor = system["traffic_predictor"]
    congestion_risk = system["congestion_risk"]
    
    data = request.json
    
    # Get time parameters
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 1)
    is_weekend = day_of_week >= 5
    
    # Get route parameters
    route = data.get('route')
    
    if not route:
        # If no route is provided, use the optimize_route endpoint logic
        depot_id = data.get('depot_id')
        stop_ids = data.get('stop_ids', [])
        
        if depot_id is None:
            depot_id = list(G.nodes())[0]
        
        if not stop_ids:
            nodes = list(G.nodes())
            if depot_id in nodes:
                nodes.remove(depot_id)
            stop_count = min(5, len(nodes))
            stop_ids = random.sample(nodes, stop_count)
        
        all_stops = [depot_id] + stop_ids
        
        if not check_graph_connectivity(G, all_stops):
            new_stops = []
            for stop in all_stops:
                if stop not in G:
                    closest_node = find_nearest_accessible_node(G, stop)
                    if closest_node:
                        new_stops.append(closest_node)
                else:
                    new_stops.append(stop)
            all_stops = new_stops
        
        traffic_multipliers = traffic_predictor.predict_for_time(hour, day_of_week)
        router.update_edge_speeds(traffic_multipliers)
        
        route = router.optimize_route(all_stops)
    
    if not route:
        return jsonify({
            "error": "Failed to generate route map",
            "success": False
        }), 400
    
    all_stops = route  # Use the route as our stops
    time_matrix = router.calculate_time_matrix(all_stops, hour=hour, is_weekend=is_weekend)
    
    # Create map
    time_period_name = f"Hour_{hour}_{'Weekend' if is_weekend else 'Weekday'}"
    
    # Based on the map_type parameter, create different map views
    map_type = data.get('map_type', 'enhanced')
    
    if map_type == 'focused':
        route_map = create_route_focused_map(G, all_stops, route, time_matrix)
    else:  # default to enhanced
        route_map = create_enhanced_route_map(G, all_stops, route, time_period_name, congestion_risk)
    
    # Save map to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        route_map.save(tmp.name)
        tmp_filename = tmp.name
    
    # Return the HTML file
    return send_file(tmp_filename, mimetype='text/html')

@app.route('/api/travel-time/predict', methods=['POST'])
def predict_travel_time():
    """Predict travel time between two nodes"""
    system = initialize_system()
    G = system["G"]
    travel_predictor = system["travel_predictor"]
    
    data = request.json
    from_node = data.get('from_node')
    to_node = data.get('to_node')
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 1)
    is_weekend = day_of_week >= 5
    
    if from_node is None or to_node is None:
        return jsonify({
            "error": "from_node and to_node are required parameters",
            "success": False
        }), 400
    
    if from_node not in G or to_node not in G:
        return jsonify({
            "error": "One or both nodes not found in the network",
            "success": False
        }), 404
    try:
        travel_time = travel_predictor.predict_travel_time(from_node, to_node, hour, is_weekend)
        
        try:
            path = nx.shortest_path(G, from_node, to_node, weight="travel_time")
            
            distance = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if 'length' in G[u][v]:
                    distance += G[u][v]['length']
            
            distance_km = distance / 1000
            
            path_coords = []
            for node in path:
                if node in G.nodes():
                    path_coords.append({
                        "lat": float(G.nodes[node].get('y', 0)),
                        "lon": float(G.nodes[node].get('x', 0))
                    })
            
            return jsonify({
                "success": True,
                "from_node": from_node,
                "to_node": to_node,
                "travel_time_minutes": float(travel_time),
                "distance_km": float(distance_km),
                "path": path,
                "path_coords": path_coords,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend
            })
        
        except nx.NetworkXNoPath:
            return jsonify({
                "success": False,
                "error": "No path found between the given nodes",
                "travel_time_minutes": float(travel_time) if travel_time != float('inf') else None
            }), 404
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get system settings and parameters"""
    return jsonify({
        "time_periods": [
            {"label": "Morning Rush (8 AM)", "hour": 8, "day_of_week": 1, "is_weekend": False},
            {"label": "Mid-day (12 PM)", "hour": 12, "day_of_week": 1, "is_weekend": False},
            {"label": "Evening Rush (6 PM)", "hour": 18, "day_of_week": 1, "is_weekend": False},
            {"label": "Night (11 PM)", "hour": 23, "day_of_week": 1, "is_weekend": False},
            {"label": "Weekend (12 PM)", "hour": 12, "day_of_week": 6, "is_weekend": True}
        ],
        "map_types": [
            {"value": "enhanced", "label": "Enhanced (Full Traffic View)"},
            {"value": "focused", "label": "Focused (Route Only)"}
        ],
        "congestion_levels": [
            {"value": 0, "label": "Low", "color": "green"},
            {"value": 1, "label": "Medium", "color": "orange"},
            {"value": 2, "label": "High", "color": "red"}
        ]
    })

@app.route('/api/network/nearby-nodes', methods=['GET'])
def get_nearby_nodes():
    """Find nodes near specified coordinates"""
    system = initialize_system()
    G = system["G"]
    
    lat = float(request.args.get('lat', 0))
    lon = float(request.args.get('lon', 0))
    radius = float(request.args.get('radius', 0.001))  # Default radius in coordinate units
    limit = int(request.args.get('limit', 10))
    
    nearby_nodes = []
    
    for node_id, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
            dist = ((data['y'] - lat)**2 + (data['x'] - lon)**2)**0.5
            if dist <= radius:
                nearby_nodes.append({
                    "id": node_id,
                    "lat": float(data['y']),
                    "lon": float(data['x']),
                    "distance": float(dist),
                    "distance_km": float(dist * 111) 
                })
    
    nearby_nodes.sort(key=lambda x: x["distance"])
    
    nearby_nodes = nearby_nodes[:limit]
    
    return jsonify({
        "success": True,
        "query": {
            "lat": lat,
            "lon": lon,
            "radius": radius
        },
        "nodes": nearby_nodes,
        "count": len(nearby_nodes)
    })

@app.route('/api/network/congestion-hotspots', methods=['GET'])
def get_congestion_hotspots():
    """Get the most congested areas in the network"""
    system = initialize_system()
    G = system["G"]
    congestion_risk = system["congestion_risk"]
    
    limit = int(request.args.get('limit', 10))
    
    high_risk_edges = [(u, v) for (u, v), risk in congestion_risk.items() if risk == 2]
    
    node_risk_count = {}
    for u, v in high_risk_edges:
        if u in node_risk_count:
            node_risk_count[u] += 1
        else:
            node_risk_count[u] = 1
            
        if v in node_risk_count:
            node_risk_count[v] += 1
        else:
            node_risk_count[v] = 1
    
    hotspot_nodes = sorted(node_risk_count.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    hotspots = []
    for node_id, risk_count in hotspot_nodes:
        if node_id in G.nodes():
            hotspots.append({
                "id": node_id,
                "lat": float(G.nodes[node_id].get('y', 0)),
                "lon": float(G.nodes[node_id].get('x', 0)),
                "high_risk_road_count": risk_count,
                "connected_roads": list(G.edges(node_id))[:5]  
            })
    
    return jsonify({
        "success": True,
        "hotspots": hotspots,
        "count": len(hotspots)
    })

@app.route('/api/system/reload', methods=['POST'])
def reload_system():
    """Force reload of the road network and models (admin use)"""
    global G, edge_ids, traffic_predictor, congestion_classifier, travel_predictor, router, congestion_risk
    
    G = None
    edge_ids = None
    traffic_predictor = None
    congestion_classifier = None
    travel_predictor = None
    router = None
    congestion_risk = None
    
    system = initialize_system()
    
    return jsonify({
        "success": True,
        "message": "System reloaded successfully",
        "network_nodes": len(system["G"].nodes()),
        "network_edges": len(system["G"].edges())
    })

@app.route('/api/user-location', methods=['POST'])
def user_location_selection():
    """Process user-selected locations for routing"""
    system = initialize_system()
    G = system["G"]
    router = system["router"]
    travel_predictor = system["travel_predictor"]
    traffic_predictor = system["traffic_predictor"]
    congestion_risk = system["congestion_risk"]
    
    data = request.json
    locations = data.get('locations', [])
    hour = data.get('hour', datetime.now().hour)
    day_of_week = data.get('day_of_week', datetime.now().weekday())
    is_weekend = day_of_week >= 5
    
    if not locations or len(locations) < 2:
        return jsonify({
            "error": "At least 2 locations are required",
            "success": False
        }), 400
    
    # Find nearest nodes to selected coordinates
    selected_nodes = []
    for loc in locations:
        lat = loc.get('lat')
        lon = loc.get('lon')
        if lat is None or lon is None:
            continue
            
        # Find nearest node
        nearest_node = None
        min_dist = float('inf')
        for node_id, data in G.nodes(data=True):
            if 'y' in data and 'x' in data:
                dist = ((data['y'] - lat)**2 + (data['x'] - lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node_id
        
        if nearest_node:
            selected_nodes.append(nearest_node)
    
    if len(selected_nodes) < 2:
        return jsonify({
            "error": "Could not map at least 2 locations to network nodes",
            "success": False
        }), 400
    
    # Update traffic conditions
    traffic_multipliers = traffic_predictor.predict_for_time(hour, day_of_week)
    router.update_edge_speeds(traffic_multipliers)
    
    route = router.optimize_route(selected_nodes)
    
    if not route:
        return jsonify({
            "error": "Failed to find a valid route between selected locations",
            "success": False
        }), 400
    
    time_matrix = router.calculate_time_matrix(selected_nodes, hour=hour, is_weekend=is_weekend)
    
    time_period_name = f"User_Selected_Route_{hour}h"
    route_map = create_enhanced_route_map(G, selected_nodes, route, time_period_name, congestion_risk)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        route_map.save(tmp.name)
        tmp_filename = tmp.name
    
    return send_file(tmp_filename, mimetype='text/html')

if __name__ == '__main__':
    initialize_system()
    app.run(debug=False, host='0.0.0.0', port=7860)