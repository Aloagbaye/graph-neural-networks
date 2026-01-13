"""
Traveling Salesman Problem (TSP) Data Generation
Creates TSP instances as graphs where nodes are cities and edges represent distances.
"""

import torch
import numpy as np
from torch_geometric.data import Data
import random
import math

class TSPDataGenerator:
    """
    Generate Traveling Salesman Problem instances for GNN training.
    
    Each TSP instance is a complete graph where:
    - Nodes: Cities with (x, y) coordinates
    - Edges: Distances between all pairs of cities
    - Task: Predict which edges are in the optimal tour
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_tsp_instance(self, num_cities=20, coord_range=(0, 100)):
        """
        Generate a single TSP instance.
        
        Args:
            num_cities: Number of cities in the TSP instance
            coord_range: Tuple (min, max) for city coordinates
        
        Returns:
            Data object with graph structure, features, and optimal tour labels
        """
        # Generate random city coordinates
        coords = np.random.uniform(
            coord_range[0], coord_range[1], 
            size=(num_cities, 2)
        )
        
        # Create complete graph (all cities connected to all cities)
        edge_list = []
        edge_attrs = []  # Edge attributes (distances)
        edge_labels = []  # Binary labels: 1 if edge is in optimal tour
        
        # Compute distances and create edges
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                # Euclidean distance
                dist = np.sqrt(
                    (coords[i, 0] - coords[j, 0])**2 + 
                    (coords[i, 1] - coords[j, 1])**2
                )
                
                # Add edge in both directions (undirected graph)
                edge_list.append([i, j])
                edge_list.append([j, i])
                
                # Edge attributes: distance, normalized distance
                edge_attrs.append([dist, dist / 100.0])  # Normalize by max possible
                edge_attrs.append([dist, dist / 100.0])
                
                # Labels will be set after computing optimal tour
                edge_labels.append(0)
                edge_labels.append(0)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Node features: coordinates, normalized coordinates
        node_features = torch.zeros(num_cities, 4, dtype=torch.float)
        for i in range(num_cities):
            node_features[i, 0] = coords[i, 0]  # x coordinate
            node_features[i, 1] = coords[i, 1]  # y coordinate
            node_features[i, 2] = coords[i, 0] / coord_range[1]  # Normalized x
            node_features[i, 3] = coords[i, 1] / coord_range[1]  # Normalized y
        
        # Compute optimal tour using nearest neighbor heuristic
        # (For larger instances, you might use exact solvers like Concorde)
        optimal_tour = self._compute_tour_nearest_neighbor(coords)
        
        # Create edge labels: 1 if edge is in optimal tour
        edge_labels = torch.zeros(edge_index.shape[1], dtype=torch.long)
        for idx in range(len(optimal_tour) - 1):
            city1 = optimal_tour[idx]
            city2 = optimal_tour[idx + 1]
            
            # Find edge indices for this pair
            mask = ((edge_index[0] == city1) & (edge_index[1] == city2)) | \
                   ((edge_index[0] == city2) & (edge_index[1] == city1))
            edge_labels[mask] = 1
        
        # Close the tour (last city to first)
        city1 = optimal_tour[-1]
        city2 = optimal_tour[0]
        mask = ((edge_index[0] == city1) & (edge_index[1] == city2)) | \
               ((edge_index[0] == city2) & (edge_index[1] == city1))
        edge_labels[mask] = 1
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=edge_labels,  # Edge labels: 1 if in optimal tour
            coords=torch.tensor(coords, dtype=torch.float),
            optimal_tour=torch.tensor(optimal_tour, dtype=torch.long),
            tour_length=self._compute_tour_length(coords, optimal_tour)
        )
    
    def _compute_tour_nearest_neighbor(self, coords):
        """
        Compute a tour using nearest neighbor heuristic.
        This is a greedy approximation - not always optimal but good enough for training.
        
        Args:
            coords: Array of city coordinates [num_cities, 2]
        
        Returns:
            List of city indices in tour order
        """
        num_cities = len(coords)
        unvisited = set(range(num_cities))
        tour = []
        
        # Start from city 0
        current = 0
        tour.append(current)
        unvisited.remove(current)
        
        # Greedily visit nearest unvisited city
        while unvisited:
            nearest = None
            nearest_dist = float('inf')
            
            for city in unvisited:
                dist = np.sqrt(
                    (coords[current, 0] - coords[city, 0])**2 +
                    (coords[current, 1] - coords[city, 1])**2
                )
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = city
            
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def _compute_tour_length(self, coords, tour):
        """Compute the total length of a tour."""
        total_length = 0.0
        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i + 1) % len(tour)]
            dist = np.sqrt(
                (coords[city1, 0] - coords[city2, 0])**2 +
                (coords[city1, 1] - coords[city2, 1])**2
            )
            total_length += dist
        return total_length
    
    def generate_multiple_instances(self, num_instances=100, num_cities=20, **kwargs):
        """
        Generate multiple TSP instances.
        
        Args:
            num_instances: Number of TSP instances to generate
            num_cities: Number of cities per instance
            **kwargs: Additional arguments for generate_tsp_instance
        
        Returns:
            List of Data objects
        """
        instances = []
        for i in range(num_instances):
            # Use different seed for each instance
            np.random.seed(self.seed + i)
            instance = self.generate_tsp_instance(num_cities=num_cities, **kwargs)
            instances.append(instance)
        return instances

def visualize_tsp_instance(data, save_path='tsp_instance.png', show_tour=True):
    """
    Visualize a TSP instance with optional optimal tour.
    """
    import matplotlib.pyplot as plt
    
    coords = data.coords.numpy()
    num_cities = len(coords)
    
    plt.figure(figsize=(10, 10))
    
    # Plot cities
    plt.scatter(coords[:, 0], coords[:, 1], 
               c='red', s=200, zorder=3, label='Cities')
    
    # Add city labels
    for i in range(num_cities):
        plt.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                    fontsize=10, ha='center', va='center',
                    color='white', weight='bold')
    
    # Plot optimal tour if available
    if show_tour and hasattr(data, 'optimal_tour'):
        tour = data.optimal_tour.numpy()
        tour_coords = coords[tour]
        tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the loop
        
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 
                'b-', linewidth=2, alpha=0.6, label='Optimal Tour')
        
        # Add tour length to title
        if hasattr(data, 'tour_length'):
            title = f'TSP Instance (Tour Length: {data.tour_length:.2f})'
        else:
            title = 'TSP Instance with Optimal Tour'
    else:
        title = 'TSP Instance'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"TSP visualization saved as '{save_path}'")
    plt.close()

if __name__ == "__main__":
    print("TSP Data Generator")
    print("="*60)
    
    # Create generator
    generator = TSPDataGenerator(seed=42)
    
    # Generate a single instance
    print("\nGenerating TSP instance...")
    data = generator.generate_tsp_instance(num_cities=15, coord_range=(0, 100))
    
    print(f"\nTSP Instance Statistics:")
    print(f"  Number of cities: {data.x.shape[0]}")
    print(f"  Features per node: {data.x.shape[1]}")
    print(f"  Total edges: {data.edge_index.shape[1]}")
    print(f"  Edges in optimal tour: {data.y.sum().item()}")
    print(f"  Tour length: {data.tour_length:.2f}")
    
    # Visualize
    print("\nVisualizing TSP instance...")
    visualize_tsp_instance(data, 'tsp_instance.png')
    
    print("\n✅ TSP data generated successfully!")
    print("Next: Build the GNN model in tsp_gnn.py")

