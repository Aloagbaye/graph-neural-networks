"""
Vehicle Routing Problem GNN Model
Graph Neural Network for predicting which edges are in optimal VRP routes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data

class VRPGNN(nn.Module):
    """
    Graph Neural Network for VRP edge prediction.
    
    The model predicts which edges should be included in the vehicle routes.
    This is a binary edge classification task.
    """
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, 
                 num_layers=4, dropout=0.3, use_gat=False):
        """
        Initialize the VRP GNN.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout probability
            use_gat: If True, use GAT instead of GCN
        """
        super(VRPGNN, self).__init__()
        
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
        
        # Demand-aware attention
        self.demand_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
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
        
        # Apply demand-aware attention
        attention_weights = self.demand_attention(x)
        x = x * attention_weights
        
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

class VRPGNNWithRouteEncoding(nn.Module):
    """
    Enhanced VRP GNN with route-level encoding.
    Learns to assign customers to routes and select edges.
    """
    
    def __init__(self, num_node_features, num_edge_features, num_routes=5,
                 hidden_dim=128, num_layers=4, dropout=0.3):
        super(VRPGNNWithRouteEncoding, self).__init__()
        
        self.num_routes = num_routes
        self.num_layers = num_layers
        
        # Node encoder with GAT
        self.node_conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=dropout)
        self.node_convs = nn.ModuleList([
            GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
            for _ in range(num_layers - 2)
        ])
        self.node_conv_final = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        
        # Route embedding
        self.route_embedding = nn.Embedding(num_routes + 1, hidden_dim)  # +1 for depot
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Route assignment predictor (for each node, predict which route)
        self.route_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_routes)
        )
        
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
        """Forward pass with route encoding."""
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
        
        # Route assignment (soft assignment)
        route_logits = self.route_predictor(x)
        route_probs = F.softmax(route_logits, dim=1)
        
        # Weighted combination of route embeddings
        route_emb = self.route_embedding.weight  # [num_routes, hidden_dim]
        node_route_emb = torch.matmul(route_probs, route_emb)  # [num_nodes, hidden_dim]
        
        # Combine node features with route encoding
        x = x + node_route_emb
        
        # Process edges
        edge_features = self.edge_encoder(edge_attr)
        
        # Get node features for edges
        row, col = edge_index
        node_i = x[row]
        node_j = x[col]
        
        # Combine and classify
        combined = torch.cat([node_i, node_j, edge_features], dim=1)
        edge_logits = self.edge_classifier(combined)
        
        return F.log_softmax(edge_logits, dim=1), route_logits

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("VRP GNN Model")
    print("="*60)
    
    # Create dummy data
    num_nodes = 21  # 1 depot + 20 customers
    num_node_features = 7
    num_edge_features = 4
    
    x = torch.randn(num_nodes, num_node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)), dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], num_edge_features)
    
    # Test GCN-based model
    print("\n1. Testing GCN-based VRP model...")
    model_gcn = VRPGNN(num_node_features, num_edge_features, use_gat=False)
    output_gcn = model_gcn(x, edge_index, edge_attr)
    print(f"   Input nodes: {x.shape}")
    print(f"   Input edges: {edge_index.shape[1]}")
    print(f"   Output shape: {output_gcn.shape}")
    print(f"   Parameters: {count_parameters(model_gcn):,}")
    
    # Test GAT-based model
    print("\n2. Testing GAT-based VRP model...")
    model_gat = VRPGNN(num_node_features, num_edge_features, use_gat=True)
    output_gat = model_gat(x, edge_index, edge_attr)
    print(f"   Input nodes: {x.shape}")
    print(f"   Input edges: {edge_index.shape[1]}")
    print(f"   Output shape: {output_gat.shape}")
    print(f"   Parameters: {count_parameters(model_gat):,}")
    
    # Test route-encoding model
    print("\n3. Testing VRP model with route encoding...")
    model_route = VRPGNNWithRouteEncoding(num_node_features, num_edge_features, num_routes=5)
    edge_out, route_out = model_route(x, edge_index, edge_attr)
    print(f"   Edge output shape: {edge_out.shape}")
    print(f"   Route output shape: {route_out.shape}")
    print(f"   Parameters: {count_parameters(model_route):,}")
    
    print("\n✅ Models created successfully!")
    print("Next: Train the model in train_vrp.py")

