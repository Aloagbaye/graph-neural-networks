"""
Step 1: Understanding Basic Graph Operations
This script demonstrates how to create and work with graphs using PyTorch Geometric.
"""

import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def create_simple_graph():
    """
    Create a simple undirected graph with 4 nodes and 4 edges.
    
    Graph structure:
        A(0) ---- B(1)
        |          |
        C(2) ---- D(3)
    """
    # Edge connectivity: [source_nodes, target_nodes]
    # For undirected graph, we need both directions
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
        [1, 0, 3, 1, 3, 2, 0, 3]   # Target nodes
    ], dtype=torch.long)
    
    # Node features: each node has 2 features
    # Example: [capacity, inventory_level]
    x = torch.tensor([
        [10.0, 5.0],   # Node 0 (A)
        [15.0, 8.0],   # Node 1 (B)
        [12.0, 3.0],   # Node 2 (C)
        [20.0, 10.0]   # Node 3 (D)
    ], dtype=torch.float)
    
    # Create graph data object
    graph = Data(x=x, edge_index=edge_index)
    
    return graph

def create_supply_chain_graph():
    """
    Create a simple supply chain graph:
    Supplier -> Warehouse -> Distribution Center -> Retailer
    """
    # Directed edges: supplier -> warehouse -> distribution -> retailer
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # Source nodes
        [1, 2, 3, 4]   # Target nodes
    ], dtype=torch.long)
    
    # Node features: [node_type, capacity, inventory, location_id]
    # node_type: 0=supplier, 1=warehouse, 2=distribution, 3=retailer
    x = torch.tensor([
        [0.0, 100.0, 80.0, 0.0],   # Supplier
        [1.0, 200.0, 150.0, 1.0],  # Warehouse
        [2.0, 150.0, 100.0, 2.0],  # Distribution Center
        [3.0, 50.0, 30.0, 3.0],    # Retailer
    ], dtype=torch.float)
    
    graph = Data(x=x, edge_index=edge_index)
    return graph

def visualize_graph(graph, title="Graph Visualization"):
    """
    Visualize a graph using NetworkX and Matplotlib.
    """
    # Convert to NetworkX format
    G = nx.Graph()
    
    # Add nodes
    for i in range(graph.x.shape[0]):
        G.add_node(i, features=graph.x[i].tolist())
    
    # Add edges
    edge_list = graph.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # Draw graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=16, font_weight='bold',
            edge_color='gray', width=2, arrows=True)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150)
    print(f"Graph visualization saved as '{title.lower().replace(' ', '_')}.png'")
    plt.show()

def print_graph_info(graph, name="Graph"):
    """
    Print information about the graph.
    """
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of features per node: {graph.x.shape[1]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    print(f"\nNode features:")
    print(graph.x)
    print(f"\nEdge connectivity:")
    print(graph.edge_index)
    print(f"{'='*50}\n")

if __name__ == "__main__":
    print("Step 1: Basic Graph Operations")
    print("=" * 50)
    
    # Create simple graph
    print("\n1. Creating a simple 4-node graph...")
    simple_graph = create_simple_graph()
    print_graph_info(simple_graph, "Simple Graph")
    
    # Create supply chain graph
    print("\n2. Creating a supply chain graph...")
    supply_chain_graph = create_supply_chain_graph()
    print_graph_info(supply_chain_graph, "Supply Chain Graph")
    
    # Visualize graphs
    print("\n3. Visualizing graphs...")
    visualize_graph(simple_graph, "Simple Graph")
    visualize_graph(supply_chain_graph, "Supply Chain Graph")
    
    print("\n✅ Step 1 Complete! You've learned how to create graphs.")
    print("Next: Learn about GCN layers in step2_gcn_layer.py")

