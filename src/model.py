import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvLayer

class GCN(nn.Module):
    """
    A full 2-Layer Graph Convolutional Network (GCN).
    """
    def __init__(self, n_features, n_hidden, n_classes, dropout_rate=0.5):
        """
        Initializes the 2-layer GCN model.
        
        Args:
            n_features (int): Number of input features per node.
            n_hidden (int): Dimension of the hidden layer.
            n_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability between layers.
        """
        super(GCN, self).__init__()
        
        # First Graph Convolution Layer: maps from input features to hidden dimension
        self.gc1 = GraphConvLayer(n_features, n_hidden)
        
        # Second Graph Convolution Layer: maps from hidden dimension to class logits
        self.gc2 = GraphConvLayer(n_hidden, n_classes)
        
        # Store dropout rate
        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        """
        Forward pass for the 2-layer GCN.
        
        Args:
            x: Node feature matrix.
            adj: Normalized sparse adjacency matrix \tilde{A}_{norm}.
            
        Returns:
            Raw logits over classes for each node.
        """
        # Apply dropout to input features to prevent initial overfitting
        x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # Layer 1: Graph Convolution -> ReLU Non-linearity
        out = self.gc1(x, adj)
        out = F.relu(out)
        
        # Apply dropout between layers to prevent overfitting
        out = F.dropout(out, self.dropout_rate, training=self.training)
        
        # Layer 2: Graph Convolution -> Raw Logits
        out = self.gc2(out, adj)
        
        # Return raw logits (no softmax) as specified by requirements
        return out
