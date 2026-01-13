"""
Training Script for VRP GNN
Trains a Graph Neural Network to predict optimal VRP route edges.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from vrp_data import VRPDataGenerator, visualize_vrp_instance
from vrp_gnn import VRPGNN, count_parameters

def split_data(instances, train_ratio=0.7, val_ratio=0.15):
    """Split VRP instances into train, validation, and test sets."""
    num_instances = len(instances)
    indices = np.random.permutation(num_instances)
    
    train_size = int(num_instances * train_ratio)
    val_size = int(num_instances * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_instances = [instances[i] for i in train_indices]
    val_instances = [instances[i] for i in val_indices]
    test_instances = [instances[i] for i in test_indices]
    
    return train_instances, val_instances, test_instances

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        
        # Compute loss with class weighting for imbalance
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds

def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, 
                weight_decay=1e-5, patience=25, device='cpu'):
    """Train the VRP GNN model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Weighted loss for class imbalance
    criterion = nn.NLLLoss()
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\nTraining Progress:")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_vrp_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_vrp_model.pt'))
    
    print("-" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    return train_losses, train_accs, val_losses, val_accs

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vrp_training_curves.png', dpi=150)
    print("Training curves saved as 'vrp_training_curves.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='vrp_confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not in Route', 'In Route'],
                yticklabels=['Not in Route', 'In Route'])
    plt.title('Confusion Matrix - Edge Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved as '{save_path}'")
    plt.close()

def visualize_predicted_routes(data, model, device, save_path='vrp_predicted_routes.png'):
    """Visualize a VRP instance with predicted routes."""
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
        pred = out.argmax(dim=1)
    
    coords = data.coords.numpy()
    num_nodes = len(coords)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: True routes
    axes[0].scatter([coords[0, 0]], [coords[0, 1]], c='black', s=300, marker='s', zorder=5, label='Depot')
    axes[0].scatter(coords[1:, 0], coords[1:, 1], c='red', s=100, alpha=0.7, zorder=3)
    
    for i in range(num_nodes):
        axes[0].annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8, ha='center', va='center')
    
    if hasattr(data, 'routes'):
        for route_idx, route in enumerate(data.routes):
            if not route:
                continue
            full_route = [0] + route + [0]
            route_coords = coords[full_route]
            axes[0].plot(route_coords[:, 0], route_coords[:, 1], 
                        c=colors[route_idx % len(colors)], linewidth=2, alpha=0.7)
    
    axes[0].set_title(f'True Routes (Distance: {data.total_distance:.2f})')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Predicted edges
    axes[1].scatter([coords[0, 0]], [coords[0, 1]], c='black', s=300, marker='s', zorder=5, label='Depot')
    axes[1].scatter(coords[1:, 0], coords[1:, 1], c='red', s=100, alpha=0.7, zorder=3)
    
    for i in range(num_nodes):
        axes[1].annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8, ha='center', va='center')
    
    # Plot predicted edges
    edge_index = data.edge_index.cpu().numpy()
    predicted_edges = pred.cpu().numpy() == 1
    
    for idx, is_in_route in enumerate(predicted_edges):
        if is_in_route:
            i, j = edge_index[:, idx]
            axes[1].plot([coords[i, 0], coords[j, 0]], 
                        [coords[i, 1], coords[j, 1]], 
                        'g-', linewidth=1.5, alpha=0.5)
    
    axes[1].set_title('Predicted Route Edges')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Predicted routes visualization saved as '{save_path}'")
    plt.close()

def main():
    """Main training function."""
    print("="*80)
    print("VRP GNN Training")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate data
    print("\n1. Generating VRP instances...")
    generator = VRPDataGenerator(seed=42)
    instances = generator.generate_multiple_instances(
        num_instances=200,
        num_customers=15,
        num_vehicles=3,
        vehicle_capacity=100,
        demand_range=(10, 30)
    )
    
    print(f"   Generated {len(instances)} VRP instances")
    print(f"   Nodes per instance: {instances[0].x.shape[0]}")
    print(f"   Features per node: {instances[0].x.shape[1]}")
    print(f"   Features per edge: {instances[0].edge_attr.shape[1]}")
    
    # Visualize one instance
    print("\n2. Visualizing sample VRP instance...")
    visualize_vrp_instance(instances[0], 'vrp_sample_instance.png')
    
    # Split data
    print("\n3. Splitting data into train/val/test sets...")
    train_instances, val_instances, test_instances = split_data(instances)
    print(f"   Train instances: {len(train_instances)}")
    print(f"   Val instances: {len(val_instances)}")
    print(f"   Test instances: {len(test_instances)}")
    
    # Create data loaders
    train_loader = DataLoader(train_instances, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_instances, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_instances, batch_size=1, shuffle=False)
    
    # Create model
    print("\n4. Creating GNN model...")
    num_node_features = instances[0].x.shape[1]
    num_edge_features = instances[0].edge_attr.shape[1]
    
    model = VRPGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        num_layers=4,
        dropout=0.3,
        use_gat=False
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"   Model parameters: {num_params:,}")
    
    # Train model
    print("\n5. Training model...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader,
        epochs=150, lr=0.001, patience=25, device=device
    )
    
    # Plot training curves
    print("\n6. Plotting training curves...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    criterion = nn.NLLLoss()
    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    print("\n8. Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    # Visualize predictions on a test instance
    print("\n9. Visualizing predictions on test instance...")
    test_instance = test_instances[0]
    visualize_predicted_routes(test_instance, model, device, 'vrp_predicted_routes.png')
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print("\nNext: Run evaluate_vrp.py for detailed analysis")

if __name__ == "__main__":
    main()

