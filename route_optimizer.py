import osmnx as ox
import numpy as np
import tensorflow as tf
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import networkx as nx
import os

ox.settings.use_cache = True  
ox.settings.log_console = True
ox.settings.timeout=180

def get_simplified_road_network():
    """Get only major roads in Chandigarh for cleaner visualization"""
    try:
        custom_filter = '["highway"~"primary|secondary|tertiary"]'
        G = ox.graph_from_place("Chandigarh, India", 
                                network_type="drive", 
                                simplify=True,
                                custom_filter=custom_filter)
        
        G = ox.add_edge_speeds(G)  
        G = ox.add_edge_travel_times(G) 
        print(f"Road Network successfully loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        sector_17_point = (30.7417, 76.7875)  
        

        G_simplified = nx.DiGraph()
        
        for node, data in G.nodes(data=True):
            if 'y' in data and 'x' in data:
                dist = ((data['y'] - sector_17_point[0])**2 + 
                        (data['x'] - sector_17_point[1])**2)**0.5
                dist_km = dist * 111  
                if dist_km <= 2:  
                    G_simplified.add_node(node, **data)
        
        for u, v, data in G.edges(data=True):
            if G_simplified.has_node(u) and G_simplified.has_node(v):
                G_simplified.add_edge(u, v, **data)
        
        print(f"Simplified road network created with {len(G_simplified.nodes)} nodes")
        
        largest_cc = max(nx.weakly_connected_components(G_simplified), key=len)
        G_simplified = G_simplified.subgraph(largest_cc).copy()
        
        print(f"Final connected graph has {len(G_simplified.nodes)} nodes and {len(G_simplified.edges)} edges")
        return G_simplified
    
    except Exception as e:
        print(f"Error obtaining road network: {str(e)}")
        return create_synthetic_network()

def create_synthetic_network():
    """Create a synthetic road network as a fallback"""
    print("Creating synthetic grid network that resembles Chandigarh's sector layout...")
    
    rows, cols = 6, 6
    G = nx.DiGraph()
    
    base_lat, base_lon = 30.72, 76.76  
    grid_spacing = 0.01  
    
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            lat_offset = random.uniform(-0.0005, 0.0005)
            lon_offset = random.uniform(-0.0005, 0.0005)
            
            G.add_node(node_id, 
                       y=base_lat + i * grid_spacing + lat_offset, 
                       x=base_lon + j * grid_spacing + lon_offset) 
    
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            
            if j < cols - 1:
                target = i * cols + (j + 1)
                length = random.uniform(900, 1100) 
                G.add_edge(node_id, target, 
                           length=length,
                           speed_kph=50.0,
                           highway="secondary",
                           travel_time=length/1000/50.0*60,
                           base_speed=50.0)
                G.add_edge(target, node_id,  
                           length=length,
                           speed_kph=50.0,
                           highway="secondary",
                           travel_time=length/1000/50.0*60,
                           base_speed=50.0)
            
            if i < rows - 1:
                target = (i + 1) * cols + j
                length = random.uniform(900, 1100)  
                G.add_edge(node_id, target, 
                           length=length,
                           speed_kph=50.0,
                           highway="secondary",
                           travel_time=length/1000/50.0*60,
                           base_speed=50.0)
                G.add_edge(target, node_id,  
                           length=length,
                           speed_kph=50.0,
                           highway="secondary",
                           travel_time=length/1000/50.0*60,
                           base_speed=50.0)
                
    for i in range(rows - 1):
        for j in range(cols - 1):
            if random.random() < 0.3:  
                node_id = i * cols + j
                target = (i + 1) * cols + (j + 1)
                length = random.uniform(1200, 1400)  
                G.add_edge(node_id, target, 
                          length=length,
                          speed_kph=40.0,
                          highway="tertiary",
                          travel_time=length/1000/40.0*60,
                          base_speed=40.0)
    
    print(f"Created synthetic grid network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G
class MLTrafficPredictor:
    def __init__(self, edge_ids, num_time_periods=24):
        self.edge_ids = edge_ids
        self.model = None
        self.train_synthetic_model(num_time_periods)
    
    def train_synthetic_model(self, num_time_periods):
        """Train a model on synthetic data with realistic patterns"""
        X_train = []  
        y_train = []  
        
        start_date = datetime(2023, 1, 1)
        for day in range(7):  
            for hour in range(24):  
                current_time = start_date + timedelta(days=day, hours=hour)
                
                time_features = [
                    hour,                           
                    current_time.weekday(),         
                    1 if hour in [7, 8, 9, 17, 18, 19] else 0, 
                    1 if current_time.weekday() < 5 else 0  
                ]
                
                edge_multipliers = []
                for e in range(len(self.edge_ids)):
                    if hour in [7, 8, 9]:  
                        base_mult = 0.7
                    elif hour in [17, 18, 19]:  
                        base_mult = 0.6
                    elif hour in [10, 11, 12, 13, 14, 15, 16]:  
                        base_mult = 0.9
                    else:  
                        base_mult = 1.2
                    
                    variation = np.random.normal(0, 0.1)
                    if current_time.weekday() >= 5:  
                        base_mult = min(1.3, base_mult + 0.2)  
                    
                    multiplier = max(0.4, min(1.5, base_mult + variation))
                    edge_multipliers.append(multiplier)
                
                X_train.append(time_features)
                y_train.append(edge_multipliers)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        print("Traffic prediction model trained on synthetic data")
    
    def predict_for_time(self, hour, day_of_week):
        """Predict traffic multipliers for a specific time"""
        time_features = [
            hour,
            day_of_week,
            1 if hour in [7, 8, 9, 17, 18, 19] else 0,  
            1 if day_of_week < 5 else 0                
            ]
        
        predictions = self.model.predict([time_features])[0]
        return predictions
    
    def predict_for_time_period(self, time_period):
        """Compatible with original interface for easy integration"""
        if time_period == 0:  
            hour, day = 8, 1  
        elif time_period == 1:  
            hour, day = 12, 1  
        elif time_period == 2: 
            hour, day = 18, 1  
        else:  
            hour, day = 23, 1 
            
        return self.predict_for_time(hour, day)

    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath, edge_ids):
        predictor = MLTrafficPredictor(edge_ids, num_time_periods=24)
        predictor.model = joblib.load(filepath)
        return predictor


class RoadCongestionClassifier:
    def __init__(self, G):
        self.G = G
        self.model = None
        self.train_synthetic_model()
    
    def train_synthetic_model(self):
        """Train a model to classify roads by congestion likelihood"""
        X_train = []
        y_train = []
        
        for u, v, data in self.G.edges(data=True):
            features = [
                data.get('length', 0) / 1000,  
                1 if data.get('highway') == 'primary' else 0,
                1 if data.get('highway') == 'secondary' else 0,
                1 if data.get('highway') == 'tertiary' else 0,
                data.get('lanes', 1) if isinstance(data.get('lanes'), int) else 1,
                1 if 'oneway' in data and data['oneway'] else 0
            ]
            
            if data.get('highway') == 'primary':
                congestion = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
            elif data.get('highway') == 'secondary':
                congestion = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            else:
                congestion = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
                
            X_train.append(features)
            y_train.append(congestion)
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        print("Road congestion classifier trained on synthetic data")
    
    def classify_roads(self):
        """Classify all roads by congestion risk"""
        congestion_risk = {}
        
        for u, v, data in self.G.edges(data=True):
            features = [
                data.get('length', 0) / 1000,
                1 if data.get('highway') == 'primary' else 0,
                1 if data.get('highway') == 'secondary' else 0,
                1 if data.get('highway') == 'tertiary' else 0,
                data.get('lanes', 1) if isinstance(data.get('lanes'), int) else 1,
                1 if 'oneway' in data and data['oneway'] else 0
            ]
            
            risk = self.model.predict([features])[0]
            congestion_risk[(u, v)] = risk
            
        return congestion_risk

    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath, G):
        classifier = RoadCongestionClassifier(G)
        classifier.model = joblib.load(filepath)
        return classifier

class TravelTimePredictor:
    def __init__(self, G):
        self.G = G
        self.model = None
        self.train_synthetic_model()
    
    def train_synthetic_model(self):
        """Train a model to predict travel times with synthetic data"""
        X_train = []
        y_train = []
        
        for u, v, data in self.G.edges(data=True):
            for scenario in range(10): 
                # Road features
                distance = data.get('length', 0) / 1000  
                base_speed = data.get('base_speed', 40.0)  
                
                hour = np.random.randint(0, 24)
                is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
                is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
                
                if is_rush_hour and not is_weekend:
                    traffic_mult = np.random.uniform(0.5, 0.8)  
                elif is_weekend:
                    traffic_mult = np.random.uniform(0.9, 1.3)  
                else:
                    traffic_mult = np.random.uniform(0.7, 1.0)  
                
                features = [
                    distance,
                    base_speed,
                    1 if data.get('highway') == 'primary' else 0,
                    1 if data.get('highway') == 'secondary' else 0,
                    1 if data.get('highway') == 'tertiary' else 0,
                    hour,
                    is_rush_hour,
                    is_weekend
                ]
                
                actual_speed = base_speed * traffic_mult
                base_time = (distance / actual_speed) * 60  
                noise = np.random.normal(0, base_time * 0.1)  
                travel_time = max(0.1, base_time + noise)
                
                X_train.append(features)
                y_train.append(travel_time)
        
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        print("Travel time prediction model trained on synthetic data")
    
    def predict_travel_time(self, u, v, hour, is_weekend=False):
        """Predict travel time between two nodes"""
        if not self.G.has_edge(u, v):
            return float('inf')
            
        data = self.G[u][v]
        distance = data.get('length', 0) / 1000
        base_speed = data.get('base_speed', 40.0)
        
        features = [
            distance,
            base_speed,
            1 if data.get('highway') == 'primary' else 0,
            1 if data.get('highway') == 'secondary' else 0,
            1 if data.get('highway') == 'tertiary' else 0,
            hour,
            1 if hour in [7, 8, 9, 17, 18, 19] else 0,  # Rush hour flag
            1 if is_weekend else 0
        ]
        
        travel_time = self.model.predict([features])[0]
        return travel_time

    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath, G):
        predictor = TravelTimePredictor(G)
        predictor.model = joblib.load(filepath)
        return predictor
    
    
class EnhancedDynamicRouter:
    def __init__(self, G, edge_ids, travel_predictor):
        self.G = G
        self.edge_ids = edge_ids
        self.travel_predictor = travel_predictor
        
    def update_edge_speeds(self, traffic_multipliers):
        """Update road network with traffic multipliers"""
        for i, (u, v) in enumerate(self.edge_ids):
            if i < len(traffic_multipliers) and self.G.has_edge(u, v):
                self.G[u][v]["current_speed"] = max(5.0, traffic_multipliers[i] * self.G[u][v]["base_speed"])
                
    def calculate_time_matrix(self, stops, hour=8, is_weekend=False):
        """Compute travel time matrix using ML prediction"""
        time_matrix = np.zeros((len(stops), len(stops)))
        
        for i, source in enumerate(stops):
            for j, target in enumerate(stops):
                if i != j:
                    try:
                        path = nx.shortest_path(self.G, source, target, weight="travel_time")
                        
                        travel_time = 0
                        for idx in range(len(path)-1):
                            u, v = path[idx], path[idx+1]
                            segment_time = self.travel_predictor.predict_travel_time(u, v, hour, is_weekend)
                            travel_time += segment_time
                            
                        time_matrix[i][j] = travel_time
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        print(f"Warning: No path found between stops {i} and {j} (nodes {source} and {target})")
                        time_matrix[i][j] = 60.0  
        
        return time_matrix
    
    def optimize_route(self, stops):
        """Solve VRP using OR-Tools"""
        time_matrix = self.calculate_time_matrix(stops)
        
        manager = pywrapcp.RoutingIndexManager(len(stops), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(time_matrix[from_node][to_node] * 1000)  
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            return self._extract_route(stops, manager, routing, solution)
        return None
    
    def _extract_route(self, stops, manager, routing, solution):
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(stops[node])
            index = solution.Value(routing.NextVar(index))
        route.append(stops[manager.IndexToNode(index)])
        return route

def check_graph_connectivity(G, stops):
    """Verify that all stops are reachable from each other"""
    problems = []
    
    for i, source in enumerate(stops):
        for j, target in enumerate(stops):
            if i != j:
                try:
                    path = nx.shortest_path(G, source, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    problems.append((i, j, source, target))
    
    if problems:
        print("WARNING: The following stop pairs are not connected in the graph:")
        for i, j, source, target in problems:
            print(f"  - Stop {i} (Node {source}) to Stop {j} (Node {target})")
        
        print("\nPossible solutions:")
        print("1. Check if these nodes exist in the graph")
        print("2. Ensure the graph is fully connected")
        print("3. Pick alternative nearby locations for problematic stops")
        
        return False
    
    return True

def find_nearest_accessible_node(G, original_node, max_distance=0.001):
    """Find the nearest node that is well-connected to the network"""
    if not G.has_node(original_node):
        print(f"Node {original_node} not found in graph!")
        return None
    
    orig_y = G.nodes[original_node]['y']
    orig_x = G.nodes[original_node]['x']
    
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    
    connected_nodes = []
    for node in largest_cc:
        if node == original_node:
            continue  
            
        node_y = G.nodes[node]['y']
        node_x = G.nodes[node]['x']
        
        dist = ((node_y - orig_y)**2 + (node_x - orig_x)**2)**0.5
        connected_nodes.append((node, dist))
    
    connected_nodes.sort(key=lambda x: x[1])
    
    for node, dist in connected_nodes:
        if dist <= max_distance:
            print(f"Found alternative for {original_node}: {node} at distance {dist*111:.2f} km")
            return node
    
    if connected_nodes:
        closest_node, dist = connected_nodes[0]
        print(f"WARNING: Closest alternative to {original_node} is {closest_node} at {dist*111:.2f} km away")
        return closest_node
    
    return None

def create_enhanced_route_map(G, stops, route, time_period_name, congestion_risk):   
    center_y = G.nodes[list(G.nodes)[0]]['y']  
    center_x = G.nodes[list(G.nodes)[0]]['x']  
    
    m = folium.Map(location=[center_y, center_x], zoom_start=13)
    
    for u, v, data in G.edges(data=True):
        u_y = G.nodes[u].get('y', 0)
        u_x = G.nodes[u].get('x', 0)
        v_y = G.nodes[v].get('y', 0)
        v_x = G.nodes[v].get('x', 0)
        
        risk = congestion_risk.get((u, v), 0)
        
        if risk == 0:
            color = "green" 
            weight = 3
        elif risk == 1:
            color = "orange" 
            weight = 4
        else:
            color = "red"  
            weight = 5
        
        popup = f"""
        <b>Road Type:</b> {data.get('highway', 'Unknown')}<br>
        <b>Current Speed:</b> {data.get('current_speed', 0):.1f} km/h<br>
        <b>Base Speed:</b> {data.get('base_speed', 0):.1f} km/h<br>
        <b>ML Congestion Risk:</b> {"Low" if risk == 0 else "Medium" if risk == 1 else "High"}<br>
        <b>Congestion:</b> {(1-data.get('current_speed', 0)/data.get('base_speed', 1))*100:.1f}%
        """
        
        folium.PolyLine(
            locations=[(u_y, u_x), (v_y, v_x)],
            color=color,
            weight=weight,
            opacity=0.7,
            popup=popup
        ).add_to(m)
    
    for i, node in enumerate(stops):
        node_y = G.nodes[node]['y']
        node_x = G.nodes[node]['x']
        
        if i == 0:  
            folium.Marker(
                location=[node_y, node_x],
                tooltip="Depot",
                popup="<b>Depot</b><br>Starting point for delivery route",
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m)
        else:  
            folium.Marker(
                location=[node_y, node_x],
                tooltip=f"Delivery Stop {i}",
                popup=f"<b>Delivery Stop {i}</b>",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
    
    if route:
        route_coords = []
        for node in route:
            point = [G.nodes[node]['y'], G.nodes[node]['x']]
            route_coords.append(point)
        
        route_line = folium.PolyLine(
            locations=route_coords,
            color="purple",
            weight=4,
            opacity=0.8,
            popup="Optimized Delivery Route"
        ).add_to(m)
        
        for i in range(len(route_coords) - 1):
            midpoint = [(route_coords[i][0] + route_coords[i+1][0]) / 2,
                       (route_coords[i][1] + route_coords[i+1][1]) / 2]
            
            folium.RegularPolygonMarker(
                location=midpoint,
                number_of_sides=3,
                radius=6,
                rotation=45,
                color="purple",
                fill_color="purple",
                fill_opacity=0.7,
                popup=f"Segment {i+1}"
            ).add_to(m)
    
    title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                   z-index:9999; font-size:18px; background-color:white; padding:10px;
                   border-radius:5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
            <h3 style="margin:0;">Optimized Delivery Route - {time_period_name}</h3>
        </div>
    '''
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; 
               background-color:white; padding:10px; border-radius:5px;
               box-shadow: 0 0 5px rgba(0,0,0,0.3); z-index:9999; font-size:12px;">
        <h4 style="margin-top:0;">Legend</h4>
        <div><span style="color:green; font-size:16px;">━━</span> Light Traffic (< 10% slowdown)</div>
        <div><span style="color:orange; font-size:16px;">━━</span> Moderate Traffic (10-30% slowdown)</div>
        <div><span style="color:red; font-size:16px;">━━</span> Heavy Traffic (> 30% slowdown)</div>
        <div><span style="color:purple; font-size:16px;">━━</span> Optimized Route</div>
        <div><span style="color:green;">●</span> Depot</div>
        <div><span style="color:blue;">●</span> Delivery Stops</div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_route_focused_map(G, stops, route, time_matrix):
    """Create a map focused just on the optimal route with time estimates"""
    center_y = G.nodes[route[0]]['y']
    center_x = G.nodes[route[0]]['x']
    
    m = folium.Map(location=[center_y, center_x], zoom_start=13)
    
    total_time = 0
    
    for i in range(len(route) - 1):
        start_node = route[i]
        end_node = route[i+1]
        
        start_idx = route.index(start_node)
        end_idx = route.index(end_node)
        
        segment_time = time_matrix[start_idx][end_idx]
        total_time += segment_time
        
        start_y = G.nodes[start_node]['y']
        start_x = G.nodes[start_node]['x']
        end_y = G.nodes[end_node]['y']
        end_x = G.nodes[end_node]['x']
        
        popup_text = f"""
        <b>Segment {i+1}:</b> Stop {i} → Stop {i+1}<br>
        <b>Travel Time:</b> {segment_time:.1f} minutes<br>
        <b>Running Total:</b> {total_time:.1f} minutes
        """
        
        folium.PolyLine(
            locations=[(start_y, start_x), (end_y, end_x)],
            color="blue",
            weight=5,
            opacity=0.8,
            popup=popup_text
        ).add_to(m)
        
        midpoint = [(start_y + end_y) / 2, (start_x + end_x) / 2]
        folium.RegularPolygonMarker(
            location=midpoint,
            number_of_sides=3,
            radius=6,
            rotation=45,
            color="blue",
            fill_color="blue",
            fill_opacity=0.7
        ).add_to(m)
    
    for i, node in enumerate(route):
        node_y = G.nodes[node]['y']
        node_x = G.nodes[node]['x']
        
        if i == 0:
            popup_text = "<b>Depot</b><br>Starting Point"
            icon = folium.Icon(color="green", icon="home")
        else:
            cumulative_time = sum(time_matrix[route.index(route[j])][route.index(route[j+1])] 
                                 for j in range(i))
            popup_text = f"""
            <b>Stop {i}</b><br>
            <b>Estimated Arrival:</b> {cumulative_time:.1f} minutes from start
            """
            icon = folium.Icon(color="red", icon="info-sign")
        
        folium.Marker(
            location=[node_y, node_x],
            tooltip=f"Stop {i}",
            popup=popup_text,
            icon=icon
        ).add_to(m)
    
    title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                   z-index:9999; font-size:18px; background-color:white; padding:10px;
                   border-radius:5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
            <h3 style="margin:0;">Optimized Delivery Route</h3>
            <div style="text-align:center; margin-top:5px; font-weight:bold;">
                Total Time: {total_time:.1f} minutes
            </div>
        </div>
    '''
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def add_base_speeds_to_graph(G):
    """Add base_speed attribute to all edges if it doesn't exist"""
    print("Adding base speeds to graph edges...")
    for u, v, data in G.edges(data=True):
        if 'speed_kph' in data:
            data['base_speed'] = data['speed_kph']
        elif 'maxspeed' in data and data['maxspeed'] is not None:
            try:
                speed_str = data['maxspeed'].split()[0]
                speed = float(speed_str)
                if 'mph' in data['maxspeed']:
                    speed = speed * 1.60934
                data['base_speed'] = speed
            except (ValueError, IndexError):
                data['base_speed'] = 40.0  
        else:
            if data.get('highway') == 'primary':
                data['base_speed'] = 50.0
            elif data.get('highway') == 'secondary':
                data['base_speed'] = 40.0
            elif data.get('highway') == 'tertiary':
                data['base_speed'] = 30.0
            else:
                data['base_speed'] = 40.0  
    
    print("Base speeds added to all edges")
    return G

def display_route_details(G, route, time_matrix):
    """Display the optimal route and estimated arrival times clearly"""
    total_time = 0
    print("\n===== OPTIMIZED DELIVERY ROUTE =====")
    print(f"Starting from depot at Node {route[0]}")
    
    for i in range(1, len(route)):
        prev_stop = route[i-1]
        curr_stop = route[i]
        
        prev_idx = route.index(prev_stop)
        curr_idx = route.index(curr_stop)
        
        segment_time = time_matrix[prev_idx][curr_idx]
        
        if segment_time > 1000:  
            print(f"\nWARNING: Unrealistic travel time detected for segment {i-1} to {i}")
            print(f"Stop {i}: Node {curr_stop}")
            print(f"  Travel time calculation error - please check connectivity")
            prev_lat = G.nodes[prev_stop]['y']
            prev_lon = G.nodes[prev_stop]['x']
            curr_lat = G.nodes[curr_stop]['y']  
            curr_lon = G.nodes[curr_stop]['x']
            
            dist_km = ((prev_lat - curr_lat)**2 + (prev_lon - curr_lon)**2)**0.5 * 111
            
            estimated_time = (dist_km / 30) * 60
            print(f"  Estimated travel time based on distance: {estimated_time:.1f} minutes")
            
            total_time += estimated_time
        else:
            total_time += segment_time
            print(f"\nStop {i}: Node {curr_stop} ({G.nodes[curr_stop]['y']:.4f}, {G.nodes[curr_stop]['x']:.4f})")
            print(f"  Travel from previous stop: {segment_time:.1f} minutes")
            print(f"  Cumulative travel time: {total_time:.1f} minutes")
    
    print(f"\nTotal route time: {total_time:.1f} minutes")
    print("==================================")
    
    return total_time

def test_end_to_end(hour=12, is_weekend=False):
    print(f"\n--- End-to-End System Test (Hour: {hour}, Weekend: {is_weekend}) ---")
    
    print("1. Loading road network...")
    G = get_simplified_road_network()
    G = add_base_speeds_to_graph(G)
    
    print("2. Setting up delivery stops...")
    nodes = list(G.nodes())
    depot = nodes[0]
    delivery_stops = random.sample(nodes[1:], min(5, len(nodes)-1))
    all_stops = [depot] + delivery_stops
    stops = all_stops  
    
    if not check_graph_connectivity(G, stops):
        print("Attempting to find alternative nodes for problematic stops...")
        new_stops = []
        
        for stop in stops:
            if stop in G:  
                cc = nx.node_connected_component(G.to_undirected(), stop)
                if len(cc) < 10:  
                    alt_node = find_nearest_accessible_node(G, stop)
                    if alt_node:
                        new_stops.append(alt_node)
                    else:
                        new_stops.append(stop)  
                else:
                    new_stops.append(stop)  
            else:
                print(f"WARNING: Stop {stop} not found in graph!")
        
        if new_stops != stops:
            print("Using alternative stops to ensure graph connectivity")
            stops = new_stops
            all_stops = stops  # Update all_stops as well
    
    print("3. Setting up ML models...")
    edge_ids = list(G.edges())
    traffic_predictor = MLTrafficPredictor(edge_ids)
    congestion_classifier = RoadCongestionClassifier(G)
    travel_predictor = TravelTimePredictor(G)
    congestion_risk = congestion_classifier.classify_roads()
    
    print("4. Predicting traffic conditions...")
    traffic_multipliers = traffic_predictor.predict_for_time(hour, 1 if not is_weekend else 6)
    
    print("5. Setting up router...")
    router = EnhancedDynamicRouter(G, edge_ids, travel_predictor)
    router.update_edge_speeds(traffic_multipliers)
    
    print("6. Optimizing route...")
    route = router.optimize_route(all_stops)
    
    time_matrix = router.calculate_time_matrix(all_stops, hour=hour, is_weekend=is_weekend)
    total_travel_time = display_route_details(G, route, time_matrix)
    route_map = create_route_focused_map(G, all_stops, route, time_matrix)
    
    if route:
        time_period_name = f"Hour_{hour}_{'Weekend' if is_weekend else 'Weekday'}"
        route_map = create_enhanced_route_map(G, all_stops, route, time_period_name, congestion_risk)
        map_filename = f"end_to_end_route_{time_period_name}.html"
        route_map.save(map_filename)
        print(f"7. Route visualization saved to {map_filename}")
        return True
    else:
        print("7. Failed to find a route")
        return False
