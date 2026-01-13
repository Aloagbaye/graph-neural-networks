# Graph Neural Networks Tutorial

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.0+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, step-by-step tutorial for learning **Graph Neural Networks (GNNs)** with hands-on optimization projects. From basic graph concepts to solving real-world logistics problems!

## 🌟 Featured Blog Posts

Learn through detailed, SEO-optimized tutorials on Hashnode:

| Topic | Blog Post | Tutorial Folder |
|-------|-----------|-----------------|
| 🗺️ **TSP** | [Solving TSP with Graph Neural Networks](blog/gnn-tsp-tutorial-hashnode.md) | `tutorials_tsp/` |
| 🚚 **VRP** | [Solving VRP with Graph Neural Networks](blog/gnn-vrp-tutorial-hashnode.md) | `tutorials_vrp/` |

---

## 📚 Overview

This repository provides a complete learning path from basic graph concepts to building and deploying GNNs for real-world problems. Includes **three complete projects**:

| # | Project | Description | Difficulty |
|---|---------|-------------|------------|
| 1 | 📦 **Supply Chain Optimization** | Predict delivery delays using node classification | ⭐ Beginner |
| 2 | 🗺️ **Traveling Salesman Problem** | Find optimal routes using edge prediction | ⭐⭐ Intermediate |
| 3 | 🚚 **Vehicle Routing Problem** | Optimize fleet routes with capacity constraints | ⭐⭐⭐ Advanced |

---

## 🎯 What You'll Learn

### Core Concepts
- **Graph Fundamentals**: Understanding nodes, edges, and graph representations
- **GNN Architecture**: How Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) work
- **Message Passing**: The core mechanism behind GNNs
- **Model Building**: Creating multi-layer GNNs for node and edge classification

### Practical Skills
- **Data Generation**: Creating synthetic training data for optimization problems
- **Training Strategies**: Handling class imbalance, early stopping, hyperparameter tuning
- **Evaluation**: Metrics, visualizations, and model interpretation
- **Real-World Applications**: Supply chain, routing, and logistics optimization

---

## 📁 Project Structure

```
graph-neural-networks/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
│
├── introduction/                     # 🎓 Basic GNN concepts
│   ├── step1_basic_graph.py         # Graph creation and visualization
│   ├── step2_gcn_layer.py           # GCN layer implementation
│   └── step3_simple_gnn.py          # Complete GNN model
│
├── delay_predictions/                # 📦 Supply Chain Project
│   ├── supply_chain_data.py         # Data generation
│   ├── supply_chain_gnn.py          # GNN model
│   ├── train_supply_chain.py        # Training script
│   └── evaluate_supply_chain.py     # Evaluation
│
├── traveling_salesman/               # 🗺️ TSP Project
│   ├── tsp_data.py                  # TSP data generation
│   ├── tsp_gnn.py                   # GNN model for TSP
│   ├── train_tsp.py                 # Training script
│   └── evaluate_tsp.py              # Evaluation
│
├── vehicle_routing_problem/          # 🚚 VRP Project
│   ├── vrp_data.py                  # VRP data generation
│   ├── vrp_gnn.py                   # GNN model for VRP
│   ├── train_vrp.py                 # Training script
│   └── evaluate_vrp.py              # Evaluation
│
├── tutorials_delay_predictions/      # 📖 Supply Chain Tutorials
│   ├── TUTORIAL.md                  # Main guide
│   ├── TUTORIAL_STEP1.md            # Graph representation
│   ├── TUTORIAL_STEP2.md            # Data generation
│   ├── TUTORIAL_STEP3.md            # GNN architecture
│   ├── TUTORIAL_STEP4.md            # Training
│   ├── TUTORIAL_STEP5.md            # Evaluation
│   └── TUTORIAL_STEP6.md            # Advanced topics
│
├── tutorials_tsp/                    # 📖 TSP Tutorials
│   ├── TUTORIAL.md                  # Main guide
│   ├── TUTORIAL_STEP1.md            # Problem definition
│   ├── TUTORIAL_STEP2.md            # Data generation
│   ├── TUTORIAL_STEP3.md            # GNN architecture
│   ├── TUTORIAL_STEP4.md            # Training
│   ├── TUTORIAL_STEP5.md            # Evaluation
│   └── TUTORIAL_STEP6.md            # Advanced topics
│
├── tutorials_vrp/                    # 📖 VRP Tutorials
│   ├── TUTORIAL.md                  # Main guide
│   ├── TUTORIAL_STEP1.md            # Problem definition
│   ├── TUTORIAL_STEP2.md            # Data generation
│   ├── TUTORIAL_STEP3.md            # GNN architecture
│   ├── TUTORIAL_STEP4.md            # Training
│   ├── TUTORIAL_STEP5.md            # Evaluation
│   └── TUTORIAL_STEP6.md            # Advanced topics
│
└── blog/                             # 📝 Blog Posts (Hashnode)
    ├── gnn-tsp-tutorial-hashnode.md # TSP blog post
    └── gnn-vrp-tutorial-hashnode.md # VRP blog post
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/Aloagbaye/graph-neural-networks.git
cd graph-neural-networks

# Install dependencies
pip install -r requirements.txt
```

### Running Projects

**1. Introduction (Learn GNN Basics)**
```bash
cd introduction
python step1_basic_graph.py    # Learn graph representation
python step2_gcn_layer.py      # Understand GCN layers
python step3_simple_gnn.py     # Build your first GNN
```

**2. Supply Chain Delay Prediction**
```bash
cd delay_predictions
python train_supply_chain.py   # Train the model
python evaluate_supply_chain.py # Evaluate performance
```

**3. Traveling Salesman Problem**
```bash
cd traveling_salesman
python train_tsp.py            # Train edge classifier
python evaluate_tsp.py         # Extract and evaluate tours
```

**4. Vehicle Routing Problem**
```bash
cd vehicle_routing_problem
python train_vrp.py            # Train with capacity constraints
python evaluate_vrp.py         # Extract and evaluate routes
```

---

## 📦 Project 1: Supply Chain Delay Prediction

**📂 Code**: `delay_predictions/` | **📖 Tutorial**: `tutorials_delay_predictions/`

### Problem
Predict which nodes in a supply chain network are at risk of delivery delays.

### Graph Structure
```
Suppliers → Warehouses → Distribution Centers → Retailers
```

### Features
| Type | Features |
|------|----------|
| **Node** | Type, capacity, inventory, processing time, reliability |
| **Task** | Node classification (delay risk vs. no delay) |
| **Model** | GCN or GAT |

### Expected Results
- Accuracy: 70-85%
- Visualizations of predictions by node type

---

## 🗺️ Project 2: Traveling Salesman Problem

**📂 Code**: `traveling_salesman/` | **📖 Tutorial**: `tutorials_tsp/` | **📝 Blog**: [`gnn-tsp-tutorial-hashnode.md`](blog/gnn-tsp-tutorial-hashnode.md)

### Problem
Find the shortest route that visits all cities exactly once and returns to the starting point.

### Graph Structure
- **Nodes**: Cities with coordinates
- **Edges**: All-pairs connections (complete graph)
- **Task**: Edge classification (predict edges in optimal tour)

### Features
| Type | Features |
|------|----------|
| **Node** | Coordinates (absolute and normalized) |
| **Edge** | Distances (absolute and normalized) |

### Expected Results
- Edge classification accuracy: 75-90%
- Tour quality: 1.2-1.5x optimal

---

## 🚚 Project 3: Vehicle Routing Problem

**📂 Code**: `vehicle_routing_problem/` | **📖 Tutorial**: `tutorials_vrp/` | **📝 Blog**: [`gnn-vrp-tutorial-hashnode.md`](blog/gnn-vrp-tutorial-hashnode.md)

### Problem
Find optimal routes for a fleet of vehicles to serve all customers while respecting capacity constraints.

### Graph Structure
- **Node 0**: Depot (start/end point)
- **Nodes 1-N**: Customers with demands
- **Task**: Edge classification (predict edges in routes)

### Features
| Type | Features |
|------|----------|
| **Node** | Coordinates, demand, normalized demand, is_depot flag |
| **Edge** | Distance, normalized distance, depot connectivity flags |

### Constraints
- ✅ Vehicle capacity limits
- ✅ All customers must be served
- ✅ Routes start and end at depot

### Expected Results
- Edge classification accuracy: 70-85%
- Route quality: 1.2-1.6x heuristic solutions

---

## 📖 Learning Path

### 🟢 Beginner Path
1. Start with `introduction/step1_basic_graph.py`
2. Work through steps 2 and 3
3. Read `tutorials_delay_predictions/TUTORIAL.md`
4. Run the supply chain project

### 🟡 Intermediate Path
1. Complete the TSP project
2. Read `tutorials_tsp/TUTORIAL.md` or the [TSP blog post](blog/gnn-tsp-tutorial-hashnode.md)
3. Experiment with hyperparameters

### 🔴 Advanced Path
1. Complete the VRP project
2. Read `tutorials_vrp/TUTORIAL.md` or the [VRP blog post](blog/gnn-vrp-tutorial-hashnode.md)
3. Compare GCN vs GAT architectures
4. Modify for your own problems

---

## 🎓 Tutorial Contents

Each project has detailed step-by-step tutorials:

| Step | Topic | Supply Chain | TSP | VRP |
|------|-------|:------------:|:---:|:---:|
| 1 | Problem & Graph Representation | ✅ | ✅ | ✅ |
| 2 | Data Generation | ✅ | ✅ | ✅ |
| 3 | GNN Architecture | ✅ | ✅ | ✅ |
| 4 | Training | ✅ | ✅ | ✅ |
| 5 | Evaluation | ✅ | ✅ | ✅ |
| 6 | Advanced Topics | ✅ | ✅ | ✅ |

---

## 📊 Project Comparison

| Aspect | Supply Chain | TSP | VRP |
|--------|:------------:|:---:|:---:|
| **Task** | Node classification | Edge classification | Edge classification |
| **Nodes** | Facilities | Cities | Depot + Customers |
| **Constraints** | None | Visit all once | Capacity limits |
| **Output** | Delay risk | Single tour | Multiple routes |
| **Algorithm** | Random labels | Nearest neighbor | Clarke-Wright |
| **Complexity** | ⭐ Easier | ⭐⭐ Medium | ⭐⭐⭐ Harder |
| **Class Balance** | ~50/50 | ~10% positive | ~7% positive |
| **Blog Post** | ❌ | ✅ | ✅ |

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| **PyTorch** | Deep learning framework |
| **PyTorch Geometric** | Graph neural network library |
| **NumPy** | Numerical computing |
| **Matplotlib** | Visualization |
| **Scikit-learn** | Machine learning utilities |
| **NetworkX** | Graph analysis |
| **Seaborn** | Statistical visualization |
| **Pandas** | Data manipulation |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📚 Additional Resources

### Documentation
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Papers
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1812.08434)
- [Understanding GCNs](https://tkipf.github.io/graph-convolutional-networks/)
- [Attention Model for VRP](https://arxiv.org/abs/1803.08475)

### Benchmarks
- [TSP Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
- [VRP Wikipedia](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [CVRPLIB Benchmark Instances](http://vrp.atd-lab.inf.puc-rio.br/)
- [Google OR-Tools](https://developers.google.com/optimization/routing)

---

## 💡 Tips for Success

1. **Start Simple**: Begin with introduction, then progress to projects
2. **Visualize**: Use visualization functions to understand model behavior
3. **Experiment**: Try different hyperparameters and architectures
4. **Compare**: GCN vs GAT - see which works better for your problem
5. **Read the Tutorials**: Each step builds on the previous one
6. **Check the Blogs**: Hashnode posts provide additional context and explanations

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share your own implementations and results

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 About the Author

Hi, I'm **Israel**, a data scientist and AI engineer passionate about transforming real-world challenges into innovative solutions with machine learning and data. I love mentoring and supporting others as they grow in their tech careers. When I'm not coding or coaching, you'll likely find me immersed in a game of chess or enjoying a good action movie with my family.

### Connect with Me

| Platform | Link |
|----------|------|
| 📝 **Hashnode** | [@israelcodes](https://hashnode.com/@israelcodes) |
| 💻 **GitHub** | [@Aloagbaye](https://github.com/Aloagbaye) |
| 💼 **LinkedIn** | [Aloagbaye](https://linkedin.com/in/Aloagbaye) |

---

## ⭐ Star This Repo

If you found this tutorial helpful, please give it a ⭐ and share it with others learning about Graph Neural Networks!

---

**Happy Learning!** 🚀

*Questions or suggestions? Feel free to open an issue or reach out on social media!*
