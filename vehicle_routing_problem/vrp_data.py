"""
Vehicle Routing Problem (VRP) Data Generation
Creates VRP instances as graphs where:
- Node 0 is the depot
- Other nodes are customers with demands
- Vehicles have capacity constraints
"""

import torch
import numpy as np
from torch_geometric.data import Data
import random
import math

class VRPDataGenerator:
    """
    Generate Vehicle Routing Problem instances for GNN training.
    
    VRP extends TSP with:
    - Multiple vehicles
    - Customer demands
    - Vehicle capacity constraints
    - Depot (start/end point)
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_vrp_instance(self, 
                              num_customers=20,
                              num_vehicles=3,
                              vehicle_capacity=100,
                              coord_range=(0, 100),
                              demand_range=(5, 25)):
        """
        Generate a single VRP instance.
        
        Args:
            num_customers: Number of customer nodes (excluding depot)
            num_vehicles: Number of available vehicles
            vehicle_capacity: Maximum capacity per vehicle
            coord_range: Tuple (min, max) for coordinates
            demand_range: Tuple (min, max) for customer demands
        
        Returns:
            Data object with graph structure, features, and route labels
        """
        num_nodes = num_customers + 1  # +1 for depot (node 0)
        
        # Generate coordinates
        # Depot at center, customers around it
        coords = np.zeros((num_nodes, 2))
        coords[0] = [coord_range[1] / 2, coord_range[1] / 2]  # Depot at center
        coords[1:] = np.random.uniform(
            coord_range[0], coord_range[1], 
            size=(num_customers, 2)
        )
        
        # Generate demands (depot has 0 demand)
        demands = np.zeros(num_nodes)
        demands[1:] = np.random.randint(
            demand_range[0], demand_range[1] + 1, 
            size=num_customers
        )
        
        # Create complete graph edges and distances
        edge_list = []
        edge_attrs = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Euclidean distance
                    dist = np.sqrt(
                        (coords[i, 0] - coords[j, 0])**2 + 
                        (coords[i, 1] - coords[j, 1])**2
                    )
                    
                    edge_list.append([i, j])
                    
                    # Edge features: distance, normalized distance, 
                    # source is depot, target is depot
                    edge_attrs.append([
                        dist,
                        dist / (coord_range[1] * np.sqrt(2)),  # Normalize
                        1.0 if i == 0 else 0.0,  # Source is depot
                        1.0 if j == 0 else 0.0   # Target is depot
                    ])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Node features: coordinates, demand, is_depot, normalized values
        node_features = torch.zeros(num_nodes, 7, dtype=torch.float)
        for i in range(num_nodes):
            node_features[i] = torch.tensor([
                coords[i, 0],                           # x coordinate
                coords[i, 1],                           # y coordinate
                coords[i, 0] / coord_range[1],          # Normalized x
                coords[i, 1] / coord_range[1],          # Normalized y
                demands[i],                             # Demand
                demands[i] / vehicle_capacity,          # Normalized demand
                1.0 if i == 0 else 0.0                  # Is depot
            ])
        
        # Compute routes using savings algorithm
        routes = self._compute_routes_savings(
            coords, demands, num_vehicles, vehicle_capacity
        )
        
        # Create edge labels: which edges are used in the routes
        edge_labels = torch.zeros(edge_index.shape[1], dtype=torch.long)
        route_assignments = torch.zeros(num_nodes, dtype=torch.long)  # Which route each node belongs to
        
        for route_idx, route in enumerate(routes):
            # Route includes depot at start and end
            full_route = [0] + route + [0]
            for k in range(len(full_route) - 1):
                city1 = full_route[k]
                city2 = full_route[k + 1]
                
                # Find edge indices
                mask = (edge_index[0] == city1) & (edge_index[1] == city2)
                edge_labels[mask] = 1
            
            # Assign route to customers
            for customer in route:
                route_assignments[customer] = route_idx + 1  # 0 is depot
        
        # Compute total distance
        total_distance = self._compute_total_distance(coords, routes)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=edge_labels,  # Edge labels: 1 if in route
            coords=torch.tensor(coords, dtype=torch.float),
            demands=torch.tensor(demands, dtype=torch.float),
            routes=routes,
            route_assignments=route_assignments,
            total_distance=total_distance,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity
        )
    
    def _compute_routes_savings(self, coords, demands, num_vehicles, capacity):
        """
        Compute routes using Clarke-Wright Savings Algorithm.
        
        This is a classic heuristic for VRP that:
        1. Starts with each customer on a separate route
        2. Merges routes based on "savings" (reduction in distance)
        3. Respects capacity constraints
        """
        num_customers = len(coords) - 1
        depot = 0
        
        # Compute distance matrix
        dist_matrix = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i != j:
                    dist_matrix[i, j] = np.sqrt(
                        (coords[i, 0] - coords[j, 0])**2 +
                        (coords[i, 1] - coords[j, 1])**2
                    )
        
        # Compute savings for all customer pairs
        savings = []
        for i in range(1, len(coords)):
            for j in range(i + 1, len(coords)):
                # Savings = d(depot, i) + d(depot, j) - d(i, j)
                save = dist_matrix[depot, i] + dist_matrix[depot, j] - dist_matrix[i, j]
                savings.append((save, i, j))
        
        # Sort by savings (descending)
        savings.sort(key=lambda x: -x[0])
        
        # Initialize: each customer is its own route
        routes = [[i] for i in range(1, len(coords))]
        route_of = {i: i - 1 for i in range(1, len(coords))}  # customer -> route index
        
        # Merge routes based on savings
        for save, i, j in savings:
            route_i = route_of.get(i)
            route_j = route_of.get(j)
            
            if route_i is None or route_j is None:
                continue
            if route_i == route_j:
                continue
            
            # Check if i and j are at the ends of their routes
            r_i = routes[route_i]
            r_j = routes[route_j]
            
            if r_i is None or r_j is None:
                continue
            
            i_at_end = (r_i[0] == i or r_i[-1] == i)
            j_at_end = (r_j[0] == j or r_j[-1] == j)
            
            if not (i_at_end and j_at_end):
                continue
            
            # Check capacity constraint
            demand_i = sum(demands[c] for c in r_i)
            demand_j = sum(demands[c] for c in r_j)
            
            if demand_i + demand_j > capacity:
                continue
            
            # Merge routes
            # Arrange so i is at end of r_i and j is at start of r_j
            if r_i[-1] != i:
                r_i = r_i[::-1]
            if r_j[0] != j:
                r_j = r_j[::-1]
            
            # Combine
            new_route = r_i + r_j
            
            # Update routes
            routes[route_i] = new_route
            routes[route_j] = None
            
            # Update route assignments
            for c in new_route:
                route_of[c] = route_i
        
        # Remove None routes and limit to num_vehicles
        routes = [r for r in routes if r is not None]
        
        # If we have more routes than vehicles, merge smallest routes
        while len(routes) > num_vehicles:
            # Sort by total demand
            routes.sort(key=lambda r: sum(demands[c] for c in r))
            
            # Try to merge smallest into others
            smallest = routes.pop(0)
            merged = False
            
            for route in routes:
                route_demand = sum(demands[c] for c in route)
                smallest_demand = sum(demands[c] for c in smallest)
                
                if route_demand + smallest_demand <= capacity:
                    route.extend(smallest)
                    merged = True
                    break
            
            if not merged:
                # Can't merge, put it back (exceed vehicle limit)
                routes.append(smallest)
                break
        
        return routes
    
    def _compute_total_distance(self, coords, routes):
        """Compute total distance of all routes."""
        total = 0.0
        depot = 0
        
        for route in routes:
            if not route:
                continue
            
            # Distance from depot to first customer
            total += np.sqrt(
                (coords[depot, 0] - coords[route[0], 0])**2 +
                (coords[depot, 1] - coords[route[0], 1])**2
            )
            
            # Distance between consecutive customers
            for i in range(len(route) - 1):
                total += np.sqrt(
                    (coords[route[i], 0] - coords[route[i+1], 0])**2 +
                    (coords[route[i], 1] - coords[route[i+1], 1])**2
                )
            
            # Distance from last customer back to depot
            total += np.sqrt(
                (coords[route[-1], 0] - coords[depot, 0])**2 +
                (coords[route[-1], 1] - coords[depot, 1])**2
            )
        
        return total
    
    def generate_multiple_instances(self, num_instances=100, **kwargs):
        """Generate multiple VRP instances."""
        instances = []
        for i in range(num_instances):
            np.random.seed(self.seed + i)
            instance = self.generate_vrp_instance(**kwargs)
            instances.append(instance)
        return instances

def visualize_vrp_instance(data, save_path='vrp_instance.png', show_routes=True):
    """Visualize a VRP instance with routes."""
    import matplotlib.pyplot as plt
    
    coords = data.coords.numpy()
    demands = data.demands.numpy()
    num_nodes = len(coords)
    
    plt.figure(figsize=(12, 10))
    
    # Color palette for routes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot depot
    plt.scatter([coords[0, 0]], [coords[0, 1]], 
               c='black', s=400, marker='s', zorder=5, label='Depot')
    plt.annotate('D', (coords[0, 0], coords[0, 1]), 
                fontsize=12, ha='center', va='center', color='white', weight='bold')
    
    # Plot customers (sized by demand)
    for i in range(1, num_nodes):
        plt.scatter([coords[i, 0]], [coords[i, 1]], 
                   c='red', s=100 + demands[i] * 5, alpha=0.7, zorder=3)
        plt.annotate(f'{i}\n({int(demands[i])})', (coords[i, 0], coords[i, 1] - 3), 
                    fontsize=8, ha='center', va='top')
    
    # Plot routes if available
    if show_routes and hasattr(data, 'routes'):
        for route_idx, route in enumerate(data.routes):
            if not route:
                continue
            
            color = colors[route_idx % len(colors)]
            
            # Full route with depot
            full_route = [0] + route + [0]
            route_coords = coords[full_route]
            
            plt.plot(route_coords[:, 0], route_coords[:, 1], 
                    c=color, linewidth=2, alpha=0.7,
                    label=f'Route {route_idx + 1}')
    
    if hasattr(data, 'total_distance'):
        title = f'VRP Instance (Total Distance: {data.total_distance:.2f})'
    else:
        title = 'VRP Instance'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"VRP visualization saved as '{save_path}'")
    plt.close()

if __name__ == "__main__":
    print("VRP Data Generator")
    print("="*60)
    
    # Create generator
    generator = VRPDataGenerator(seed=42)
    
    # Generate a single instance
    print("\nGenerating VRP instance...")
    data = generator.generate_vrp_instance(
        num_customers=20,
        num_vehicles=4,
        vehicle_capacity=100,
        coord_range=(0, 100),
        demand_range=(10, 30)
    )
    
    print(f"\nVRP Instance Statistics:")
    print(f"  Total nodes: {data.x.shape[0]} (1 depot + {data.x.shape[0]-1} customers)")
    print(f"  Features per node: {data.x.shape[1]}")
    print(f"  Total edges: {data.edge_index.shape[1]}")
    print(f"  Edges in routes: {data.y.sum().item()}")
    print(f"  Number of routes: {len(data.routes)}")
    print(f"  Total distance: {data.total_distance:.2f}")
    print(f"  Total demand: {data.demands.sum().item():.0f}")
    print(f"  Vehicle capacity: {data.vehicle_capacity}")
    
    for i, route in enumerate(data.routes):
        route_demand = sum(data.demands[c].item() for c in route)
        print(f"  Route {i+1}: {len(route)} customers, demand: {route_demand:.0f}")
    
    # Visualize
    print("\nVisualizing VRP instance...")
    visualize_vrp_instance(data, 'vrp_instance.png')
    
    print("\n✅ VRP data generated successfully!")
    print("Next: Build the GNN model in vrp_gnn.py")

