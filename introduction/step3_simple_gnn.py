"""
Step 3: Building a Complete GNN Model
This script shows how to build a multi-layer GNN for node classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np

class SimpleGNN(nn.Module):
    """
    A simple 2-layer Graph Neural Network for node classification.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(SimpleGNN, self).__init__()
        
        # Layer 1: Input features -> Hidden dimension
        self.conv1 = GCNConv(num_features, hidden_dim)
        
        # Layer 2: Hidden dimension -> Output classes
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        # Layer 1: First convolution + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2: Second convolution (output layer)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def create_toy_dataset():
    """
    Create a toy dataset for node classification.
    
    We'll create a graph where nodes belong to 2 classes:
    - Class 0: Nodes with feature sum < 1.0
    - Class 1: Nodes with feature sum >= 1.0
    """
    # Create a graph with 20 nodes
    num_nodes = 20
    
    # Random node features (2 features per node)
    x = torch.randn(num_nodes, 2)
    
    # Create edges: connect each node to its 3 nearest neighbors (by feature similarity)
    edge_list = []
    for i in range(num_nodes):
        # Compute distances to all other nodes
        distances = torch.norm(x - x[i], dim=1)
        # Get 3 nearest neighbors (excluding self)
        _, neighbors = torch.topk(distances, k=4, largest=False)
        for neighbor in neighbors[1:]:  # Skip self
            edge_list.append([i, neighbor.item()])
            edge_list.append([neighbor.item(), i])  # Undirected
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create labels based on feature sum
    node_sums = x.sum(dim=1)
    y = (node_sums >= 0.0).long()  # Binary classification
    
    return Data(x=x, edge_index=edge_index, y=y)

def train_model(model, data, epochs=200, lr=0.01):
    """
    Train the GNN model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    # Use all nodes for training (in practice, you'd split train/val/test)
    train_mask = torch.ones(data.x.size(0), dtype=torch.bool)
    
    losses = []
    accuracies = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss
        loss = criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        pred = out.argmax(dim=1)
        acc = (pred[train_mask] == data.y[train_mask]).float().mean()
        
        losses.append(loss.item())
        accuracies.append(acc.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')
    
    return losses, accuracies

def visualize_results(data, model, losses, accuracies):
    """
    Visualize training results and predictions.
    """
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Accuracy curve
    axes[1].plot(accuracies)
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    # Node classification visualization
    # Use first two features for 2D visualization
    node_features = data.x.numpy()
    true_labels = data.y.numpy()
    pred_labels = pred.numpy()
    
    axes[2].scatter(node_features[:, 0], node_features[:, 1], 
                   c=true_labels, cmap='viridis', s=100, alpha=0.6, label='True')
    axes[2].scatter(node_features[:, 0], node_features[:, 1], 
                   c=pred_labels, cmap='viridis', s=50, alpha=0.3, marker='x', label='Pred')
    axes[2].set_title('Node Classification')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('gnn_training_results.png', dpi=150)
    print("\nTraining results saved as 'gnn_training_results.png'")
    plt.show()
    
    # Print accuracy
    accuracy = (pred == data.y).float().mean()
    print(f"\nFinal Accuracy: {accuracy.item():.4f}")

def demonstrate_gnn():
    """
    Main function to demonstrate GNN training.
    """
    print("Step 3: Building a Complete GNN Model")
    print("="*60)
    
    # Create dataset
    print("\n1. Creating toy dataset...")
    data = create_toy_dataset()
    print(f"   Nodes: {data.x.shape[0]}")
    print(f"   Features per node: {data.x.shape[1]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   Classes: {data.y.max().item() + 1}")
    
    # Create model
    print("\n2. Creating GNN model...")
    model = SimpleGNN(
        num_features=data.x.shape[1],
        hidden_dim=16,
        num_classes=2,
        dropout=0.5
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n3. Training model...")
    losses, accuracies = train_model(model, data, epochs=200, lr=0.01)
    
    # Visualize results
    print("\n4. Visualizing results...")
    visualize_results(data, model, losses, accuracies)
    
    print("\n✅ Step 3 Complete! You've built and trained a GNN.")
    print("Next: Apply GNNs to the supply chain project!")

if __name__ == "__main__":
    demonstrate_gnn()

