"""
Evaluation Script for VRP GNN
Provides detailed analysis of model predictions and route quality.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from torch_geometric.data import DataLoader

from vrp_data import VRPDataGenerator, visualize_vrp_instance
from vrp_gnn import VRPGNN

def load_model_and_data():
    """Load the trained model and generate test data."""
    # Generate data
    generator = VRPDataGenerator(seed=42)
    instances = generator.generate_multiple_instances(
        num_instances=50,
        num_customers=15,
        num_vehicles=3,
        vehicle_capacity=100,
        demand_range=(10, 30)
    )
    
    # Create model
    num_node_features = instances[0].x.shape[1]
    num_edge_features = instances[0].edge_attr.shape[1]
    
    model = VRPGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        num_layers=4,
        dropout=0.3,
        use_gat=False
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_vrp_model.pt'))
        print("✅ Loaded trained model weights")
    except FileNotFoundError:
        print("⚠️  No trained model found. Please run train_vrp.py first.")
        print("   Using untrained model for demonstration...")
    
    return model, instances

def extract_routes_from_predictions(data, pred_edges, max_routes=5):
    """
    Extract VRP routes from edge predictions.
    Uses greedy approach respecting capacity constraints.
    """
    coords = data.coords.cpu().numpy()
    demands = data.demands.cpu().numpy()
    capacity = data.vehicle_capacity
    edge_index = data.edge_index.cpu().numpy()
    
    num_nodes = len(coords)
    
    # Get edges predicted to be in routes
    route_edges = set()
    for idx, is_in_route in enumerate(pred_edges):
        if is_in_route:
            i, j = edge_index[:, idx]
            route_edges.add((i, j))
    
    # Build adjacency list from predicted edges
    adj = {i: [] for i in range(num_nodes)}
    for i, j in route_edges:
        adj[i].append(j)
    
    # Extract routes starting from depot (node 0)
    routes = []
    visited = set([0])  # Depot always "visited"
    
    for start_neighbor in adj[0]:
        if start_neighbor in visited or start_neighbor == 0:
            continue
        
        # Build route
        route = [start_neighbor]
        visited.add(start_neighbor)
        current = start_neighbor
        route_demand = demands[start_neighbor]
        
        while True:
            # Find next unvisited customer connected to current
            next_customer = None
            for neighbor in adj.get(current, []):
                if neighbor not in visited and neighbor != 0:
                    if route_demand + demands[neighbor] <= capacity:
                        next_customer = neighbor
                        break
            
            if next_customer is None:
                break
            
            route.append(next_customer)
            visited.add(next_customer)
            route_demand += demands[next_customer]
            current = next_customer
        
        if route:
            routes.append(route)
        
        if len(routes) >= max_routes:
            break
    
    # Add any remaining customers to routes if possible
    remaining = [i for i in range(1, num_nodes) if i not in visited]
    for customer in remaining:
        # Try to add to existing route
        added = False
        for route in routes:
            route_demand = sum(demands[c] for c in route)
            if route_demand + demands[customer] <= capacity:
                route.append(customer)
                visited.add(customer)
                added = True
                break
        
        if not added and len(routes) < max_routes:
            # Create new route
            routes.append([customer])
            visited.add(customer)
    
    return routes

def compute_route_distance(coords, routes):
    """Compute total distance of all routes."""
    total = 0.0
    depot = 0
    
    for route in routes:
        if not route:
            continue
        
        # Depot to first customer
        total += np.sqrt(
            (coords[depot, 0] - coords[route[0], 0])**2 +
            (coords[depot, 1] - coords[route[0], 1])**2
        )
        
        # Between customers
        for i in range(len(route) - 1):
            total += np.sqrt(
                (coords[route[i], 0] - coords[route[i+1], 0])**2 +
                (coords[route[i], 1] - coords[route[i+1], 1])**2
            )
        
        # Last customer to depot
        total += np.sqrt(
            (coords[route[-1], 0] - coords[depot, 0])**2 +
            (coords[route[-1], 1] - coords[depot, 1])**2
        )
    
    return total

def analyze_predictions(model, instances, device):
    """Analyze model predictions in detail."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    pred_distances = []
    true_distances = []
    
    with torch.no_grad():
        for data in instances:
            data_device = data.to(device)
            out = model(data_device.x, data_device.edge_index, data_device.edge_attr)
            probs = torch.exp(out)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            # Extract and evaluate routes
            pred_edges = pred.cpu().numpy() == 1
            predicted_routes = extract_routes_from_predictions(data, pred_edges)
            
            coords = data.coords.cpu().numpy()
            if predicted_routes:
                pred_dist = compute_route_distance(coords, predicted_routes)
                pred_distances.append(pred_dist)
            else:
                pred_distances.append(float('inf'))
            
            true_distances.append(data.total_distance)
    
    return (np.array(all_preds), np.array(all_labels), np.array(all_probs),
            np.array(pred_distances), np.array(true_distances))

def plot_route_quality(pred_distances, true_distances):
    """Plot comparison of predicted vs true route distances."""
    # Filter out invalid routes
    valid_mask = pred_distances != float('inf')
    valid_pred = pred_distances[valid_mask]
    valid_true = np.array(true_distances)[valid_mask]
    
    if len(valid_pred) == 0:
        print("No valid routes found in predictions.")
        return
    
    # Compute approximation ratios
    ratios = valid_pred / valid_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(valid_true, valid_pred, alpha=0.6)
    axes[0].plot([valid_true.min(), valid_true.max()], 
                [valid_true.min(), valid_true.max()], 
                'r--', label='Optimal (y=x)')
    axes[0].set_xlabel('True Route Distance', fontsize=12)
    axes[0].set_ylabel('Predicted Route Distance', fontsize=12)
    axes[0].set_title('Predicted vs True Route Distances', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(ratios.mean(), color='red', linestyle='--', 
                   label=f'Mean: {ratios.mean():.2f}')
    axes[1].set_xlabel('Approximation Ratio (Predicted / True)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Route Quality Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vrp_route_quality.png', dpi=150)
    print("Route quality analysis saved as 'vrp_route_quality.png'")
    plt.close()
    
    print(f"\nRoute Quality Statistics:")
    print(f"  Valid routes: {len(valid_pred)}/{len(pred_distances)}")
    print(f"  Mean approximation ratio: {ratios.mean():.2f}")
    print(f"  Best ratio: {ratios.min():.2f}")
    print(f"  Worst ratio: {ratios.max():.2f}")

def plot_probability_analysis(y_true, y_probs):
    """Plot probability distribution and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability distribution
    not_in_route_probs = y_probs[y_true == 0]
    in_route_probs = y_probs[y_true == 1]
    
    axes[0].hist(not_in_route_probs, bins=30, alpha=0.7, label='Not in Route', color='blue')
    axes[0].hist(in_route_probs, bins=30, alpha=0.7, label='In Route', color='red')
    axes[0].set_xlabel('Predicted Probability of Being in Route', fontsize=12)
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
    plt.savefig('vrp_probability_analysis.png', dpi=150)
    print("Probability analysis saved as 'vrp_probability_analysis.png'")
    plt.close()
    
    return roc_auc

def visualize_sample_predictions(model, instances, device, num_samples=3):
    """Visualize predictions on sample instances."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx in range(min(num_samples, len(instances))):
        data = instances[idx].to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
        
        coords = data.coords.cpu().numpy()
        num_nodes = len(coords)
        
        # True routes
        axes[idx, 0].scatter([coords[0, 0]], [coords[0, 1]], c='black', s=200, marker='s', zorder=5)
        axes[idx, 0].scatter(coords[1:, 0], coords[1:, 1], c='red', s=100, alpha=0.7, zorder=3)
        
        for i in range(num_nodes):
            axes[idx, 0].annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8, ha='center', va='center')
        
        if hasattr(data, 'routes'):
            for route_idx, route in enumerate(data.routes):
                if not route:
                    continue
                full_route = [0] + route + [0]
                route_coords = coords[full_route]
                axes[idx, 0].plot(route_coords[:, 0], route_coords[:, 1], 
                                c=colors[route_idx % len(colors)], linewidth=2, alpha=0.7)
        
        axes[idx, 0].set_title(f'True Routes (Distance: {data.total_distance:.2f})')
        axes[idx, 0].set_xlabel('X Coordinate')
        axes[idx, 0].set_ylabel('Y Coordinate')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].axis('equal')
        
        # Predicted edges
        axes[idx, 1].scatter([coords[0, 0]], [coords[0, 1]], c='black', s=200, marker='s', zorder=5)
        axes[idx, 1].scatter(coords[1:, 0], coords[1:, 1], c='red', s=100, alpha=0.7, zorder=3)
        
        for i in range(num_nodes):
            axes[idx, 1].annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=8, ha='center', va='center')
        
        edge_index = data.edge_index.cpu().numpy()
        pred_edges = pred.cpu().numpy() == 1
        
        for edge_idx, is_in_route in enumerate(pred_edges):
            if is_in_route:
                i, j = edge_index[:, edge_idx]
                axes[idx, 1].plot([coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                'g-', linewidth=1.5, alpha=0.5)
        
        # Compute predicted route distance
        predicted_routes = extract_routes_from_predictions(data, pred_edges)
        if predicted_routes:
            pred_dist = compute_route_distance(coords.cpu().numpy() if torch.is_tensor(coords) else coords, predicted_routes)
            axes[idx, 1].set_title(f'Predicted Routes (Distance: {pred_dist:.2f})')
        else:
            axes[idx, 1].set_title('Predicted Routes (Invalid)')
        
        axes[idx, 1].set_xlabel('X Coordinate')
        axes[idx, 1].set_ylabel('Y Coordinate')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('vrp_sample_predictions.png', dpi=150)
    print("Sample predictions saved as 'vrp_sample_predictions.png'")
    plt.close()

def analyze_by_demand(instances, y_true, y_pred):
    """Analyze prediction performance by node demand."""
    print("\n" + "="*60)
    print("Analysis by Demand Level")
    print("="*60)
    
    # This is a simplified analysis - in practice you'd track this per-edge
    demands = []
    for inst in instances:
        demands.extend(inst.demands.numpy()[1:])  # Exclude depot
    
    demands = np.array(demands)
    low_demand = demands < 15
    high_demand = demands >= 25
    
    print(f"\nDemand distribution:")
    print(f"  Low demand (<15): {low_demand.sum()} nodes")
    print(f"  High demand (>=25): {high_demand.sum()} nodes")
    print(f"  Average demand: {demands.mean():.2f}")

def main():
    """Main evaluation function."""
    print("="*80)
    print("VRP GNN Evaluation")
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
    y_pred, y_true, y_probs, pred_distances, true_distances = analyze_predictions(
        model, instances, device
    )
    
    # Classification report
    print("\n3. Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred,
                               target_names=['Not in Route', 'In Route']))
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Edge Classification Accuracy: {accuracy:.4f}")
    
    # Route quality analysis
    print("\n4. Analyzing route quality...")
    plot_route_quality(pred_distances, true_distances)
    
    # Probability analysis
    print("\n5. Analyzing prediction probabilities...")
    roc_auc = plot_probability_analysis(y_true, y_probs)
    print(f"   ROC AUC Score: {roc_auc:.4f}")
    
    # Visualize sample predictions
    print("\n6. Visualizing sample predictions...")
    visualize_sample_predictions(model, instances, device, num_samples=3)
    
    # Analyze by demand
    print("\n7. Analyzing by demand level...")
    analyze_by_demand(instances, y_true, y_pred)
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - vrp_route_quality.png")
    print("  - vrp_probability_analysis.png")
    print("  - vrp_sample_predictions.png")

if __name__ == "__main__":
    main()

