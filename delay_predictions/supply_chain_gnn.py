"""
Supply Chain GNN Model
Graph Neural Network for predicting delivery delays in supply chains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

class SupplyChainGNN(nn.Module):
    """
    Graph Neural Network for supply chain delay prediction.
    
    Uses Graph Convolutional Networks (GCN) to learn node representations
    and predict which nodes are at risk of delays.
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2, num_layers=3, 
                 dropout=0.5, use_gat=False):
        """
        Initialize the Supply Chain GNN.
        
        Args:
            num_features: Number of input features per node
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes (2 for binary classification)
            num_layers: Number of GNN layers
            dropout: Dropout probability
            use_gat: If True, use Graph Attention Networks instead of GCN
        """
        super(SupplyChainGNN, self).__init__()
        
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        # Input layer
        if use_gat:
            self.conv1 = GATConv(num_features, hidden_dim, heads=4, dropout=dropout)
            self.convs = nn.ModuleList([
                GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
                for _ in range(num_layers - 2)
            ])
            self.conv_final = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        else:
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers - 2)
            ])
            self.conv_final = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for graph-level tasks (optional)
        
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        # First layer
        if self.use_gat:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            if self.use_gat:
                x = conv(x, edge_index)
                x = F.elu(x)
            else:
                x = conv(x, edge_index)
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.conv_final(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

class SupplyChainGNNWithAttention(nn.Module):
    """
    Enhanced GNN with attention mechanism for supply chain prediction.
    Uses Graph Attention Networks (GAT) which can learn different importance
    weights for different neighbors.
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2, num_layers=3, 
                 dropout=0.5, num_heads=4):
        super(SupplyChainGNNWithAttention, self).__init__()
        
        self.num_layers = num_layers
        
        # Multi-head attention layers
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        
        self.convs = nn.ModuleList([
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 2)
        ])
        
        # Final layer: single head
        self.conv_final = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass with attention mechanism."""
        # First layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.conv_final(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Supply Chain GNN Model")
    print("="*60)
    
    # Create a dummy graph for testing
    num_nodes = 29  # 5 suppliers + 8 warehouses + 6 DCs + 10 retailers
    num_features = 5
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 50), dtype=torch.long)
    
    # Test GCN-based model
    print("\n1. Testing GCN-based model...")
    model_gcn = SupplyChainGNN(num_features=num_features, use_gat=False)
    output_gcn = model_gcn(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_gcn.shape}")
    print(f"   Parameters: {count_parameters(model_gcn):,}")
    
    # Test GAT-based model
    print("\n2. Testing GAT-based model...")
    model_gat = SupplyChainGNN(num_features=num_features, use_gat=True)
    output_gat = model_gat(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_gat.shape}")
    print(f"   Parameters: {count_parameters(model_gat):,}")
    
    print("\n✅ Models created successfully!")
    print("Next: Train the model in train_supply_chain.py")

