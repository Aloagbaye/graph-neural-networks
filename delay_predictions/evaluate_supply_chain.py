"""
Evaluation Script for Supply Chain GNN
Provides detailed analysis and visualization of model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from torch_geometric.data import Data

from supply_chain_data import SupplyChainDataGenerator, visualize_supply_chain
from supply_chain_gnn import SupplyChainGNN

def load_model_and_data():
    """Load the trained model and generate test data."""
    # Generate data (same seed as training)
    generator = SupplyChainDataGenerator(seed=42)
    data = generator.generate_supply_chain(
        num_suppliers=5,
        num_warehouses=8,
        num_distribution_centers=6,
        num_retailers=10
    )
    
    # Create model
    model = SupplyChainGNN(
        num_features=data.x.shape[1],
        hidden_dim=64,
        num_classes=2,
        num_layers=3,
        dropout=0.5,
        use_gat=False
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_model.pt'))
        print("✅ Loaded trained model weights")
    except FileNotFoundError:
        print("⚠️  No trained model found. Please run train_supply_chain.py first.")
        print("   Using untrained model for demonstration...")
    
    return model, data

def analyze_predictions(model, data):
    """Analyze model predictions in detail."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)  # Convert log probabilities to probabilities
        pred = out.argmax(dim=1)
    
    # Convert to numpy
    y_true = data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_probs = probs[:, 1].cpu().numpy()  # Probability of delay risk
    
    return y_true, y_pred, y_probs

def plot_node_predictions(data, y_true, y_pred, y_probs):
    """Visualize predictions on the graph structure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get node positions (using feature space)
    node_features = data.x.cpu().numpy()
    
    # Plot 1: True labels
    scatter1 = axes[0].scatter(node_features[:, 1], node_features[:, 2], 
                              c=y_true, cmap='RdYlGn', s=100, alpha=0.7,
                              edgecolors='black', linewidths=1)
    axes[0].set_xlabel('Capacity', fontsize=12)
    axes[0].set_ylabel('Current Inventory', fontsize=12)
    axes[0].set_title('True Labels (Delay Risk)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Delay Risk (0=No, 1=Yes)')
    
    # Plot 2: Predictions
    scatter2 = axes[1].scatter(node_features[:, 1], node_features[:, 2],
                              c=y_pred, cmap='RdYlGn', s=100, alpha=0.7,
                              edgecolors='black', linewidths=1)
    axes[1].set_xlabel('Capacity', fontsize=12)
    axes[1].set_ylabel('Current Inventory', fontsize=12)
    axes[1].set_title('Model Predictions', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Delay Risk (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('node_predictions.png', dpi=150)
    print("Node predictions visualization saved as 'node_predictions.png'")
    plt.close()

def plot_probability_distribution(y_true, y_probs):
    """Plot the distribution of predicted probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability distribution by class
    no_delay_probs = y_probs[y_true == 0]
    delay_probs = y_probs[y_true == 1]
    
    axes[0].hist(no_delay_probs, bins=20, alpha=0.7, label='No Delay', color='green')
    axes[0].hist(delay_probs, bins=20, alpha=0.7, label='Delay Risk', color='red')
    axes[0].set_xlabel('Predicted Probability of Delay Risk', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Probability Distribution by True Class', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_analysis.png', dpi=150)
    print("Probability analysis saved as 'probability_analysis.png'")
    plt.close()
    
    return roc_auc

def analyze_by_node_type(data, y_true, y_pred):
    """Analyze predictions by node type."""
    node_types = data.x[:, 0].cpu().numpy().astype(int)
    type_names = ['Supplier', 'Warehouse', 'Distribution', 'Retailer']
    
    print("\n" + "="*60)
    print("Analysis by Node Type")
    print("="*60)
    
    for node_type in range(4):
        mask = node_types == node_type
        if mask.sum() == 0:
            continue
        
        type_true = y_true[mask]
        type_pred = y_pred[mask]
        
        accuracy = (type_true == type_pred).mean()
        delay_rate = type_true.mean()
        pred_delay_rate = type_pred.mean()
        
        print(f"\n{type_names[node_type]}:")
        print(f"  Count: {mask.sum()}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  True delay rate: {delay_rate:.4f}")
        print(f"  Predicted delay rate: {pred_delay_rate:.4f}")

def plot_feature_importance_analysis(data, y_true, y_pred):
    """Analyze which features are most important for predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    feature_names = ['Node Type', 'Capacity', 'Inventory', 'Processing Time', 'Reliability']
    
    for i, feature_name in enumerate(feature_names):
        feature_values = data.x[:, i].cpu().numpy()
        
        # Separate by prediction correctness
        correct_mask = (y_true == y_pred)
        incorrect_mask = ~correct_mask
        
        axes[i].scatter(feature_values[correct_mask], y_pred[correct_mask],
                       alpha=0.6, label='Correct', s=50, color='green')
        axes[i].scatter(feature_values[incorrect_mask], y_pred[incorrect_mask],
                       alpha=0.6, label='Incorrect', s=50, color='red', marker='x')
        axes[i].set_xlabel(feature_name, fontsize=10)
        axes[i].set_ylabel('Predicted Class', fontsize=10)
        axes[i].set_title(f'{feature_name} vs Predictions', fontsize=11, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=150)
    print("Feature analysis saved as 'feature_analysis.png'")
    plt.close()

def main():
    """Main evaluation function."""
    print("="*80)
    print("Supply Chain GNN Evaluation")
    print("="*80)
    
    # Load model and data
    print("\n1. Loading model and data...")
    model, data = load_model_and_data()
    
    # Analyze predictions
    print("\n2. Analyzing predictions...")
    y_true, y_pred, y_probs = analyze_predictions(model, data)
    
    # Classification report
    print("\n3. Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                               target_names=['No Delay', 'Delay Risk']))
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Plot node predictions
    print("\n4. Visualizing node predictions...")
    plot_node_predictions(data, y_true, y_pred, y_probs)
    
    # Probability analysis
    print("\n5. Analyzing prediction probabilities...")
    roc_auc = plot_probability_distribution(y_true, y_probs)
    print(f"   ROC AUC Score: {roc_auc:.4f}")
    
    # Analysis by node type
    print("\n6. Analyzing by node type...")
    analyze_by_node_type(data, y_true, y_pred)
    
    # Feature analysis
    print("\n7. Analyzing feature importance...")
    plot_feature_importance_analysis(data, y_true, y_pred)
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - node_predictions.png")
    print("  - probability_analysis.png")
    print("  - feature_analysis.png")

if __name__ == "__main__":
    main()

