"""
Traveling Salesman Problem GNN Model
Graph Neural Network for predicting which edges are in the optimal TSP tour.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data

class TSPGNN(nn.Module):
    """
    Graph Neural Network for TSP edge prediction.
    
    The model predicts which edges should be included in the optimal tour.
    This is a binary edge classification task.
    """
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, 
                 num_layers=3, dropout=0.3, use_gat=False):
        """
        Initialize the TSP GNN.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout probability
            use_gat: If True, use GAT instead of GCN
        """
        super(TSPGNN, self).__init__()
        
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        # Node feature encoder
        if use_gat:
            self.node_conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=dropout)
            self.node_convs = nn.ModuleList([
                GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
                for _ in range(num_layers - 2)
            ])
            self.node_conv_final = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        else:
            self.node_conv1 = GCNConv(num_node_features, hidden_dim)
            self.node_convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers - 2)
            ])
            self.node_conv_final = GCNConv(hidden_dim, hidden_dim)
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge classifier: combines node and edge features
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 2 nodes + 1 edge
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
        
        Returns:
            Edge predictions [num_edges, 2] (log probabilities)
        """
        # Process node features through GNN layers
        if self.use_gat:
            x = self.node_conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            for conv in self.node_convs:
                x = conv(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.node_conv_final(x, edge_index)
            x = F.relu(x)
        else:
            x = self.node_conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            for conv in self.node_convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.node_conv_final(x, edge_index)
            x = F.relu(x)
        
        # Process edge features
        edge_features = self.edge_encoder(edge_attr)
        
        # Get node features for each edge
        row, col = edge_index
        node_features_i = x[row]  # Source node features
        node_features_j = x[col]  # Target node features
        
        # Combine node and edge features
        edge_combined = torch.cat([
            node_features_i,
            node_features_j,
            edge_features
        ], dim=1)
        
        # Classify edges
        edge_logits = self.edge_classifier(edge_combined)
        
        return F.log_softmax(edge_logits, dim=1)

class TSPGNNWithAttention(nn.Module):
    """
    Enhanced TSP GNN with attention mechanism for edge prediction.
    """
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128,
                 num_layers=3, dropout=0.3, num_heads=4):
        super(TSPGNNWithAttention, self).__init__()
        
        self.num_layers = num_layers
        
        # Multi-head attention for nodes
        self.node_conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.node_convs = nn.ModuleList([
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            for _ in range(num_layers - 2)
        ])
        self.node_conv_final = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Edge feature encoder with attention
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for combining node and edge features
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass with attention mechanism."""
        # Process nodes
        x = self.node_conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for conv in self.node_convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.node_conv_final(x, edge_index)
        x = F.relu(x)
        
        # Process edges
        edge_features = self.edge_encoder(edge_attr)
        
        # Get node features for edges
        row, col = edge_index
        node_i = x[row].unsqueeze(1)  # [num_edges, 1, hidden_dim]
        node_j = x[col].unsqueeze(1)  # [num_edges, 1, hidden_dim]
        edge_feat = edge_features.unsqueeze(1)  # [num_edges, 1, hidden_dim]
        
        # Combine and apply attention
        combined = torch.cat([node_i, node_j, edge_feat], dim=1)  # [num_edges, 3, hidden_dim]
        attended, _ = self.attention(combined, combined, combined)
        attended = attended.mean(dim=1)  # [num_edges, hidden_dim]
        
        # Final classification
        edge_combined = torch.cat([
            x[row],
            x[col],
            edge_features
        ], dim=1)
        
        edge_logits = self.edge_classifier(edge_combined)
        
        return F.log_softmax(edge_logits, dim=1)

def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("TSP GNN Model")
    print("="*60)
    
    # Create dummy data
    num_cities = 15
    num_node_features = 4
    num_edge_features = 2
    
    x = torch.randn(num_cities, num_node_features)
    edge_index = torch.randint(0, num_cities, (2, num_cities * (num_cities - 1)), dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], num_edge_features)
    
    # Test GCN-based model
    print("\n1. Testing GCN-based model...")
    model_gcn = TSPGNN(num_node_features, num_edge_features, use_gat=False)
    output_gcn = model_gcn(x, edge_index, edge_attr)
    print(f"   Input nodes: {x.shape}")
    print(f"   Input edges: {edge_index.shape[1]}")
    print(f"   Output shape: {output_gcn.shape}")
    print(f"   Parameters: {count_parameters(model_gcn):,}")
    
    # Test GAT-based model
    print("\n2. Testing GAT-based model...")
    model_gat = TSPGNN(num_node_features, num_edge_features, use_gat=True)
    output_gat = model_gat(x, edge_index, edge_attr)
    print(f"   Input nodes: {x.shape}")
    print(f"   Input edges: {edge_index.shape[1]}")
    print(f"   Output shape: {output_gat.shape}")
    print(f"   Parameters: {count_parameters(model_gat):,}")
    
    print("\n✅ Models created successfully!")
    print("Next: Train the model in train_tsp.py")

