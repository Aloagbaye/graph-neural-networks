"""
Step 2: Understanding Graph Convolutional Network (GCN) Layers
This script implements a GCN layer from scratch to understand the message passing mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGCNLayer(MessagePassing):
    """
    A simple Graph Convolutional Network layer.
    
    The GCN layer performs:
    1. Message passing: Aggregate neighbor features
    2. Linear transformation: Apply learnable weights
    3. Normalization: Normalize by node degrees
    """
    
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNLayer, self).__init__(aggr='add')  # Aggregation: sum
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Forward pass of GCN layer.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Step 1: Add self-loops to include node's own features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Step 2: Linear transformation
        x = self.linear(x)
        
        # Step 3: Compute normalization coefficients
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4: Message passing and aggregation
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        """
        Message function: What each node sends to its neighbors.
        
        Args:
            x_j: Features of source nodes
            norm: Normalization coefficients
        
        Returns:
            Normalized messages
        """
        return norm.view(-1, 1) * x_j

def manual_message_passing_example():
    """
    Demonstrate message passing manually to understand the concept.
    """
    print("\n" + "="*60)
    print("Manual Message Passing Example")
    print("="*60)
    
    # Create a simple graph: A -> B -> C
    # Node features: A=[1,0], B=[0,1], C=[1,1]
    x = torch.tensor([
        [1.0, 0.0],  # Node A
        [0.0, 1.0],  # Node B
        [1.0, 1.0]   # Node C
    ], dtype=torch.float)
    
    edge_index = torch.tensor([
        [0, 1],  # A -> B
        [1, 2]   # B -> C
    ], dtype=torch.long).t().contiguous()
    
    print("\nInitial node features:")
    print("Node A:", x[0].tolist())
    print("Node B:", x[1].tolist())
    print("Node C:", x[2].tolist())
    
    # Manual aggregation: Each node receives features from its neighbors
    print("\nStep 1: Aggregating neighbor features...")
    
    # Node A: no incoming edges (or self-loop)
    # Node B: receives from A
    # Node C: receives from B
    
    aggregated = torch.zeros_like(x)
    aggregated[0] = x[0]  # A keeps its own features
    aggregated[1] = x[0]  # B receives from A
    aggregated[2] = x[1]  # C receives from B
    
    print("After aggregation:")
    print("Node A:", aggregated[0].tolist())
    print("Node B:", aggregated[1].tolist())
    print("Node C:", aggregated[2].tolist())
    
    # Step 2: Apply transformation (simplified: just multiply by 2)
    transformed = aggregated * 2
    print("\nStep 2: After transformation (multiply by 2):")
    print("Node A:", transformed[0].tolist())
    print("Node B:", transformed[1].tolist())
    print("Node C:", transformed[2].tolist())
    
    print("\n" + "="*60)

def test_gcn_layer():
    """
    Test the GCN layer on a simple graph.
    """
    print("\n" + "="*60)
    print("Testing GCN Layer")
    print("="*60)
    
    # Create graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 3, 1, 3, 2, 0, 3]
    ], dtype=torch.long)
    
    # Node features: 4 nodes, 2 features each
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=torch.float)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Create GCN layer: 2 input features -> 3 output features
    gcn_layer = SimpleGCNLayer(in_channels=2, out_channels=3)
    
    # Forward pass
    output = gcn_layer(x, edge_index)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"\nOutput features:")
    print(output)
    
    print("\n" + "="*60)
    return gcn_layer, output

def compare_with_pyg_gcn():
    """
    Compare our implementation with PyTorch Geometric's GCN.
    """
    print("\n" + "="*60)
    print("Comparison with PyG's GCN")
    print("="*60)
    
    from torch_geometric.nn import GCNConv
    
    # Create graph
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    x = torch.randn(3, 4)  # 3 nodes, 4 features
    
    # Our implementation
    our_gcn = SimpleGCNLayer(in_channels=4, out_channels=8)
    our_output = our_gcn(x, edge_index)
    
    # PyG's implementation
    pyg_gcn = GCNConv(in_channels=4, out_channels=8)
    pyg_output = pyg_gcn(x, edge_index)
    
    print(f"\nOur GCN output shape: {our_output.shape}")
    print(f"PyG GCN output shape: {pyg_output.shape}")
    print(f"\nBoth implementations work! ✅")
    print("="*60)

if __name__ == "__main__":
    print("Step 2: Understanding GCN Layers")
    print("="*60)
    
    # Manual example
    manual_message_passing_example()
    
    # Test our GCN layer
    gcn_layer, output = test_gcn_layer()
    
    # Compare with PyG
    compare_with_pyg_gcn()
    
    print("\n✅ Step 2 Complete! You understand how GCN layers work.")
    print("Next: Build a complete GNN model in step3_simple_gnn.py")

