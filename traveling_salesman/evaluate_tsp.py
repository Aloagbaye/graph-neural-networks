"""
Evaluation Script for TSP GNN
Provides detailed analysis of model predictions and tour quality.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from torch_geometric.data import DataLoader

from tsp_data import TSPDataGenerator, visualize_tsp_instance
from tsp_gnn import TSPGNN

def load_model_and_data():
    """Load the trained model and generate test data."""
    # Generate data (same seed as training)
    generator = TSPDataGenerator(seed=42)
    instances = generator.generate_multiple_instances(
        num_instances=50,
        num_cities=15,
        coord_range=(0, 100)
    )
    
    # Create model
    num_node_features = instances[0].x.shape[1]
    num_edge_features = instances[0].edge_attr.shape[1]
    
    model = TSPGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        use_gat=False
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_tsp_model.pt'))
        print("✅ Loaded trained model weights")
    except FileNotFoundError:
        print("⚠️  No trained model found. Please run train_tsp.py first.")
        print("   Using untrained model for demonstration...")
    
    return model, instances

def extract_tour_from_predictions(data, pred_edges):
    """
    Extract a tour from edge predictions.
    Uses a simple greedy approach to build a valid tour.
    """
    num_cities = data.x.shape[0]
    edge_index = data.edge_index.cpu().numpy()
    
    # Get edges predicted to be in tour
    tour_edges = []
    for idx, is_in_tour in enumerate(pred_edges):
        if is_in_tour:
            i, j = edge_index[:, idx]
            tour_edges.append((i, j))
    
    # Build tour using greedy approach
    if len(tour_edges) == 0:
        return None
    
    # Start from city 0
    tour = [0]
    visited = {0}
    current = 0
    
    while len(tour) < num_cities:
        # Find next city connected to current
        next_city = None
        for i, j in tour_edges:
            if i == current and j not in visited:
                next_city = j
                break
            elif j == current and i not in visited:
                next_city = i
                break
        
        if next_city is None:
            # If no valid connection, find nearest unvisited city
            unvisited = [c for c in range(num_cities) if c not in visited]
            if unvisited:
                next_city = unvisited[0]
            else:
                break
        
        tour.append(next_city)
        visited.add(next_city)
        current = next_city
    
    return tour if len(tour) == num_cities else None

def compute_tour_length(coords, tour):
    """Compute the total length of a tour."""
    if tour is None or len(tour) != len(coords):
        return float('inf')
    
    total_length = 0.0
    for i in range(len(tour)):
        city1 = tour[i]
        city2 = tour[(i + 1) % len(tour)]
        dist = np.sqrt(
            (coords[city1, 0] - coords[city2, 0])**2 +
            (coords[city1, 1] - coords[city2, 1])**2
        )
        total_length += dist
    return total_length

def analyze_predictions(model, instances, device):
    """Analyze model predictions in detail."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    tour_lengths = []
    optimal_lengths = []
    
    with torch.no_grad():
        for data in instances:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            probs = torch.exp(out)  # Convert log probabilities
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being in tour
            
            # Extract and evaluate tour
            pred_edges = pred.cpu().numpy() == 1
            predicted_tour = extract_tour_from_predictions(data, pred_edges)
            
            coords = data.coords.cpu().numpy()
            if predicted_tour:
                pred_length = compute_tour_length(coords, predicted_tour)
                tour_lengths.append(pred_length)
            else:
                tour_lengths.append(float('inf'))
            
            if hasattr(data, 'tour_length'):
                optimal_lengths.append(data.tour_length.item())
    
    return (np.array(all_preds), np.array(all_labels), np.array(all_probs),
            np.array(tour_lengths), np.array(optimal_lengths))

def plot_tour_quality(tour_lengths, optimal_lengths):
    """Plot comparison of predicted vs optimal tour lengths."""
    # Filter out invalid tours
    valid_mask = tour_lengths != float('inf')
    valid_pred = tour_lengths[valid_mask]
    valid_opt = optimal_lengths[valid_mask]
    
    if len(valid_pred) == 0:
        print("No valid tours found in predictions.")
        return
    
    # Compute approximation ratios
    ratios = valid_pred / valid_opt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: predicted vs optimal
    axes[0].scatter(valid_opt, valid_pred, alpha=0.6)
    axes[0].plot([valid_opt.min(), valid_opt.max()], 
                [valid_opt.min(), valid_opt.max()], 
                'r--', label='Optimal (y=x)')
    axes[0].set_xlabel('Optimal Tour Length', fontsize=12)
    axes[0].set_ylabel('Predicted Tour Length', fontsize=12)
    axes[0].set_title('Predicted vs Optimal Tour Lengths', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of approximation ratios
    axes[1].hist(ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(ratios.mean(), color='red', linestyle='--', 
                   label=f'Mean: {ratios.mean():.2f}')
    axes[1].set_xlabel('Approximation Ratio (Predicted / Optimal)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Tour Quality Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_tour_quality.png', dpi=150)
    print("Tour quality analysis saved as 'tsp_tour_quality.png'")
    plt.close()
    
    print(f"\nTour Quality Statistics:")
    print(f"  Valid tours: {len(valid_pred)}/{len(tour_lengths)}")
    print(f"  Mean approximation ratio: {ratios.mean():.2f}")
    print(f"  Best ratio: {ratios.min():.2f}")
    print(f"  Worst ratio: {ratios.max():.2f}")

def plot_probability_analysis(y_true, y_probs):
    """Plot probability distribution and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability distribution
    not_in_tour_probs = y_probs[y_true == 0]
    in_tour_probs = y_probs[y_true == 1]
    
    axes[0].hist(not_in_tour_probs, bins=30, alpha=0.7, label='Not in Tour', color='blue')
    axes[0].hist(in_tour_probs, bins=30, alpha=0.7, label='In Tour', color='red')
    axes[0].set_xlabel('Predicted Probability of Being in Tour', fontsize=12)
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
    plt.savefig('tsp_probability_analysis.png', dpi=150)
    print("Probability analysis saved as 'tsp_probability_analysis.png'")
    plt.close()
    
    return roc_auc

def visualize_sample_predictions(model, instances, device, num_samples=3):
    """Visualize predictions on sample instances."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(min(num_samples, len(instances))):
        data = instances[idx].to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
        
        coords = data.coords.cpu().numpy()
        num_cities = len(coords)
        
        # True tour
        axes[idx, 0].scatter(coords[:, 0], coords[:, 1], c='red', s=200, zorder=3)
        for i in range(num_cities):
            axes[idx, 0].annotate(str(i), (coords[i, 0], coords[i, 1]),
                                fontsize=8, ha='center', va='center', color='white', weight='bold')
        
        if hasattr(data, 'optimal_tour'):
            tour = data.optimal_tour.cpu().numpy()
            tour_coords = coords[tour]
            tour_coords = np.vstack([tour_coords, tour_coords[0]])
            axes[idx, 0].plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.6)
            if hasattr(data, 'tour_length'):
                axes[idx, 0].set_title(f'True Tour (Length: {data.tour_length:.2f})')
        
        axes[idx, 0].set_xlabel('X Coordinate')
        axes[idx, 0].set_ylabel('Y Coordinate')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].axis('equal')
        
        # Predicted tour
        axes[idx, 1].scatter(coords[:, 0], coords[:, 1], c='red', s=200, zorder=3)
        for i in range(num_cities):
            axes[idx, 1].annotate(str(i), (coords[i, 0], coords[i, 1]),
                                fontsize=8, ha='center', va='center', color='white', weight='bold')
        
        edge_index = data.edge_index.cpu().numpy()
        pred_edges = pred.cpu().numpy() == 1
        
        for edge_idx, is_in_tour in enumerate(pred_edges):
            if is_in_tour:
                i, j = edge_index[:, edge_idx]
                axes[idx, 1].plot([coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                'g-', linewidth=1, alpha=0.4)
        
        # Compute predicted tour length
        predicted_tour = extract_tour_from_predictions(data, pred_edges)
        if predicted_tour:
            pred_length = compute_tour_length(coords, predicted_tour)
            axes[idx, 1].set_title(f'Predicted Tour (Length: {pred_length:.2f})')
        else:
            axes[idx, 1].set_title('Predicted Tour (Invalid)')
        
        axes[idx, 1].set_xlabel('X Coordinate')
        axes[idx, 1].set_ylabel('Y Coordinate')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('tsp_sample_predictions.png', dpi=150)
    print("Sample predictions saved as 'tsp_sample_predictions.png'")
    plt.close()

def main():
    """Main evaluation function."""
    print("="*80)
    print("TSP GNN Evaluation")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model and data
    print("\n1. Loading model and data...")
    model, instances = load_model_and_data()
    model = model.to(device)
    
    # Analyze predictions
    print("\n2. Analyzing predictions...")
    y_pred, y_true, y_probs, tour_lengths, optimal_lengths = analyze_predictions(
        model, instances, device
    )
    
    # Classification report
    print("\n3. Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred,
                               target_names=['Not in Tour', 'In Tour']))
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Edge Classification Accuracy: {accuracy:.4f}")
    
    # Tour quality analysis
    print("\n4. Analyzing tour quality...")
    plot_tour_quality(tour_lengths, optimal_lengths)
    
    # Probability analysis
    print("\n5. Analyzing prediction probabilities...")
    roc_auc = plot_probability_analysis(y_true, y_probs)
    print(f"   ROC AUC Score: {roc_auc:.4f}")
    
    # Visualize sample predictions
    print("\n6. Visualizing sample predictions...")
    visualize_sample_predictions(model, instances, device, num_samples=3)
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - tsp_tour_quality.png")
    print("  - tsp_probability_analysis.png")
    print("  - tsp_sample_predictions.png")

if __name__ == "__main__":
    main()

