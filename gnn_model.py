"""
Graph Neural Network (GNN) architecture for chemical density prediction.
Treats the 4 molecular features as nodes in a graph and learns relationships.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data
import numpy as np
from typing import Tuple


class GraphNeuralSurrogate(nn.Module):
    """GNN-based surrogate model for chemical density prediction.
    
    Treats input features as a fully-connected graph where:
    - 4 nodes represent [SigC, SigH, EpsC, EpsH]
    - Edges connect all pairs (fully connected graph)
    - GNN learns to aggregate and process node features
    
    Architecture options:
    - GCN (Graph Convolutional Network): Standard message passing
    - GAT (Graph Attention Network): Attention-based message passing
    - GraphConv: Generic graph convolution
    """
    
    def __init__(
        self,
        num_node_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        """Initialize GNN surrogate model.
        
        Args:
            num_node_features: Number of input features (default 4 for SigC, SigH, EpsC, EpsH)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ("gcn", "gat", or "graphconv")
            dropout_rate: Dropout rate for regularization
            output_dim: Output dimension (1 for density prediction)
        """
        super().__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(num_node_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            if self.gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif self.gnn_type == "graphconv":
                self.gnn_layers.append(GraphConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")
            
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Global mean pooling + output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def _create_graph(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a fully connected graph for the 4 features.
        
        Returns:
            edge_index: Tensor of shape (2, num_edges) defining edge connections
            batch: Tensor of shape (batch_size * 4,) defining which graph each node belongs to
        """
        num_nodes = self.num_node_features
        
        # Create fully connected edge list
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops (can add with self_loops=True if desired)
                    edges.append([i, j])
        
        if len(edges) == 0:
            # Add self-loops if no edges
            edges = [[i, i] for i in range(num_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        
        # Create batch tensor (which graph does each node belong to)
        batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        
        return edge_index, batch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            x: Input tensor of shape (batch_size, 4) with features [SigC, SigH, EpsC, EpsH]
        
        Returns:
            Output tensor of shape (batch_size, 1) with density predictions
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Project input features to hidden dimension
        # Shape: (batch_size, 4) -> (batch_size, 4, hidden_dim)
        x_proj = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # Reshape for graph: treat each feature as a node
        # We need to expand to have multiple nodes per sample
        # Create node features: (batch_size * num_nodes, hidden_dim)
        x_nodes = x_proj.unsqueeze(1).expand(-1, self.num_node_features, -1)
        x_nodes = x_nodes.reshape(batch_size * self.num_node_features, self.hidden_dim)
        
        # Create edge indices for fully connected graph
        edge_index_single, _ = self._create_graph(1, device)
        
        # Replicate edge indices for all samples in batch
        edge_index_batch = []
        for b in range(batch_size):
            edge_index_batch.append(edge_index_single + b * self.num_node_features)
        edge_index = torch.cat(edge_index_batch, dim=1)
        
        # Apply GNN layers
        x_gnn = x_nodes
        for gnn_layer, dropout_layer in zip(self.gnn_layers, self.dropout_layers):
            x_gnn = gnn_layer(x_gnn, edge_index)
            x_gnn = torch.relu(x_gnn)
            x_gnn = dropout_layer(x_gnn)
        
        # Global mean pooling: average node embeddings for each sample
        x_pooled = torch.zeros(batch_size, self.hidden_dim, device=device)
        for b in range(batch_size):
            start_idx = b * self.num_node_features
            end_idx = (b + 1) * self.num_node_features
            x_pooled[b] = x_gnn[start_idx:end_idx].mean(dim=0)
        
        # Output projection
        output = self.output_projection(x_pooled)
        
        return output
    
    def get_model_info(self) -> str:
        """Get a summary of the model architecture."""
        info_lines = [
            "=" * 60,
            "Graph Neural Network (GNN) Surrogate Model",
            "=" * 60,
            f"Input features: {self.num_node_features} (nodes in graph)",
            f"Graph type: Fully connected (all-to-all edges)",
            f"GNN type: {self.gnn_type.upper()}",
            f"Hidden dimension: {self.hidden_dim}",
            f"Number of GNN layers: {self.num_layers}",
            f"Dropout rate: {self.dropout_rate}",
            f"Output dimension: {self.output_dim}",
            f"Total parameters: {sum(p.numel() for p in self.parameters()):,}",
            "=" * 60,
        ]
        return "\n".join(info_lines)
