"""
Training Script for Supply Chain GNN
Trains a Graph Neural Network to predict delivery delays in supply chains.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from supply_chain_data import SupplyChainDataGenerator, visualize_supply_chain
from supply_chain_gnn import SupplyChainGNN, SupplyChainGNNWithAttention

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split nodes into train, validation, and test sets.
    
    Args:
        data: PyTorch Geometric Data object
        train_ratio: Ratio of nodes for training
        val_ratio: Ratio of nodes for validation
    
    Returns:
        train_mask, val_mask, test_mask
    """
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask

def train_epoch(model, data, train_mask, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    pred = out.argmax(dim=1)
    acc = (pred[train_mask] == data.y[train_mask]).float().mean()
    
    return loss.item(), acc.item()

def evaluate(model, data, mask):
    """Evaluate the model."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        loss = nn.NLLLoss()(out[mask], data.y[mask])
        acc = (pred[mask] == data.y[mask]).float().mean()
        
        # Additional metrics
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return loss.item(), acc.item(), precision, recall, f1, y_true, y_pred

def train_model(model, data, train_mask, val_mask, test_mask, 
                epochs=200, lr=0.01, weight_decay=5e-4, patience=30):
    """
    Train the GNN model with early stopping.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        train_loss, train_acc = train_epoch(model, data, train_mask, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _, _, _, _ = evaluate(model, data, val_mask)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
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
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved as 'training_curves.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Delay', 'Delay Risk'],
                yticklabels=['No Delay', 'Delay Risk'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved as '{save_path}'")
    plt.close()

def main():
    """Main training function."""
    print("="*80)
    print("Supply Chain GNN Training")
    print("="*80)
    
    # Generate data
    print("\n1. Generating supply chain data...")
    generator = SupplyChainDataGenerator(seed=42)
    data = generator.generate_supply_chain(
        num_suppliers=5,
        num_warehouses=8,
        num_distribution_centers=6,
        num_retailers=10
    )
    
    print(f"   Nodes: {data.x.shape[0]}")
    print(f"   Features: {data.x.shape[1]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   Delay risk nodes: {data.y.sum().item()}")
    print(f"   Normal nodes: {(data.y == 0).sum().item()}")
    
    # Visualize graph
    print("\n2. Visualizing supply chain graph...")
    visualize_supply_chain(data, 'supply_chain_graph.png')
    
    # Split data
    print("\n3. Splitting data into train/val/test sets...")
    train_mask, val_mask, test_mask = split_data(data, train_ratio=0.7, val_ratio=0.15)
    print(f"   Train nodes: {train_mask.sum().item()}")
    print(f"   Val nodes: {val_mask.sum().item()}")
    print(f"   Test nodes: {test_mask.sum().item()}")
    
    # Create model
    print("\n4. Creating GNN model...")
    model = SupplyChainGNN(
        num_features=data.x.shape[1],
        hidden_dim=64,
        num_classes=2,
        num_layers=3,
        dropout=0.5,
        use_gat=False  # Set to True to use Graph Attention Networks
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Train model
    print("\n5. Training model...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, data, train_mask, val_mask, test_mask,
        epochs=200, lr=0.01, patience=30
    )
    
    # Plot training curves
    print("\n6. Plotting training curves...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred = evaluate(
        model, data, test_mask
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
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print("\nNext: Run evaluate_supply_chain.py for detailed analysis")

if __name__ == "__main__":
    main()

