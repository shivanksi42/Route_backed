import os
import sys

if not os.path.exists('route_optimizer.py'):
    print("Error: route_optimizer.py file not found!")
    print("Please save your original code as 'route_optimizer.py' before running this application.")
    sys.exit(1)

from app import app

if __name__ == '__main__':
    print("Starting Route Optimization API Server...")
    
    from app import initialize_system
    print("Initializing route optimization system...")
    system = initialize_system()
    print(f"System initialized with {len(system['G'].nodes())} nodes and {len(system['G'].edges())} edges")
    
    app.run(debug=True, host='0.0.0.0', port=5000)