"""
Supply Chain Data Generation
Creates synthetic supply chain graphs with nodes representing different entities
and edges representing shipping routes.
"""

import torch
import numpy as np
from torch_geometric.data import Data
import random

class SupplyChainDataGenerator:
    """
    Generate synthetic supply chain graphs for training GNNs.
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_supply_chain(self, 
                            num_suppliers=5,
                            num_warehouses=8,
                            num_distribution_centers=6,
                            num_retailers=10,
                            feature_dim=5):
        """
        Generate a supply chain graph.
        
        Args:
            num_suppliers: Number of supplier nodes
            num_warehouses: Number of warehouse nodes
            num_distribution_centers: Number of distribution center nodes
            num_retailers: Number of retailer nodes
            feature_dim: Number of features per node
        
        Returns:
            Data object with graph structure and features
        """
        # Calculate total nodes
        total_nodes = num_suppliers + num_warehouses + num_distribution_centers + num_retailers
        
        # Node type mapping
        node_types = []
        node_types.extend([0] * num_suppliers)  # 0 = supplier
        node_types.extend([1] * num_warehouses)  # 1 = warehouse
        node_types.extend([2] * num_distribution_centers)  # 2 = distribution center
        node_types.extend([3] * num_retailers)  # 3 = retailer
        
        # Create edges (supply chain flow)
        edge_list = []
        
        # Suppliers -> Warehouses
        for s in range(num_suppliers):
            # Each supplier connects to 2-3 random warehouses
            num_connections = random.randint(2, 3)
            warehouses = random.sample(range(num_suppliers, num_suppliers + num_warehouses), 
                                      num_connections)
            for w in warehouses:
                edge_list.append([s, w])
        
        # Warehouses -> Distribution Centers
        for w in range(num_suppliers, num_suppliers + num_warehouses):
            # Each warehouse connects to 1-2 distribution centers
            num_connections = random.randint(1, 2)
            dist_centers = random.sample(
                range(num_suppliers + num_warehouses, 
                      num_suppliers + num_warehouses + num_distribution_centers),
                num_connections
            )
            for dc in dist_centers:
                edge_list.append([w, dc])
        
        # Distribution Centers -> Retailers
        for dc in range(num_suppliers + num_warehouses, 
                       num_suppliers + num_warehouses + num_distribution_centers):
            # Each distribution center connects to 2-4 retailers
            num_connections = random.randint(2, 4)
            retailers = random.sample(
                range(num_suppliers + num_warehouses + num_distribution_centers, total_nodes),
                min(num_connections, num_retailers)
            )
            for r in retailers:
                edge_list.append([dc, r])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Generate node features
        # Features: [node_type, capacity, current_inventory, processing_time, reliability_score]
        x = torch.zeros(total_nodes, feature_dim)
        
        for i in range(total_nodes):
            node_type = node_types[i]
            x[i, 0] = float(node_type)  # Node type
            
            # Capacity (varies by node type)
            if node_type == 0:  # Supplier
                capacity = random.uniform(100, 200)
            elif node_type == 1:  # Warehouse
                capacity = random.uniform(200, 400)
            elif node_type == 2:  # Distribution Center
                capacity = random.uniform(150, 300)
            else:  # Retailer
                capacity = random.uniform(50, 150)
            
            x[i, 1] = capacity
            
            # Current inventory (percentage of capacity)
            x[i, 2] = random.uniform(0.3, 0.9) * capacity
            
            # Processing time (hours)
            x[i, 3] = random.uniform(1.0, 24.0)
            
            # Reliability score (0-1)
            x[i, 4] = random.uniform(0.6, 1.0)
        
        # Generate labels: predict delay risk (binary classification)
        # Delay risk = 1 if inventory < 0.4 * capacity OR reliability < 0.7
        y = torch.zeros(total_nodes, dtype=torch.long)
        for i in range(total_nodes):
            inventory_ratio = x[i, 2] / x[i, 1] if x[i, 1] > 0 else 0
            if inventory_ratio < 0.4 or x[i, 4] < 0.7:
                y[i] = 1  # High delay risk
        
        return Data(x=x, edge_index=edge_index, y=y, node_types=torch.tensor(node_types))
    
    def generate_multiple_graphs(self, num_graphs=10, **kwargs):
        """
        Generate multiple supply chain graphs.
        
        Args:
            num_graphs: Number of graphs to generate
            **kwargs: Arguments to pass to generate_supply_chain
        
        Returns:
            List of Data objects
        """
        graphs = []
        for i in range(num_graphs):
            graph = self.generate_supply_chain(**kwargs)
            graphs.append(graph)
        return graphs

def visualize_supply_chain(data, save_path='supply_chain_graph.png'):
    """
    Visualize the supply chain graph structure.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with colors based on type
    node_colors = []
    type_names = ['Supplier', 'Warehouse', 'Distribution', 'Retailer']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(data.x.shape[0]):
        node_type = int(data.x[i, 0])
        G.add_node(i, node_type=node_type)
        node_colors.append(colors[node_type])
    
    # Add edges
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # Draw graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    for node_type, color in enumerate(colors):
        nodes = [i for i in range(data.x.shape[0]) if int(data.x[i, 0]) == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=color, node_size=500, 
                              alpha=0.8, label=type_names[node_type])
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Supply Chain Graph Structure', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Supply chain graph saved as '{save_path}'")
    plt.close()

if __name__ == "__main__":
    print("Supply Chain Data Generator")
    print("="*60)
    
    # Create generator
    generator = SupplyChainDataGenerator(seed=42)
    
    # Generate a single graph
    print("\nGenerating supply chain graph...")
    data = generator.generate_supply_chain(
        num_suppliers=5,
        num_warehouses=8,
        num_distribution_centers=6,
        num_retailers=10
    )
    
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {data.x.shape[0]}")
    print(f"  Features per node: {data.x.shape[1]}")
    print(f"  Total edges: {data.edge_index.shape[1]}")
    print(f"  Nodes with delay risk: {data.y.sum().item()}")
    print(f"  Nodes without delay risk: {(data.y == 0).sum().item()}")
    
    # Visualize
    print("\nVisualizing graph...")
    visualize_supply_chain(data)
    
    print("\n✅ Supply chain data generated successfully!")
    print("Next: Build the GNN model in supply_chain_gnn.py")

