import math
import torch
import torch.nn as nn

# GCN Update Rule:
# H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})
class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer from scratch (Kipf & Welling, 2017).
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the Graph Convolutional Layer.
        
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            bias (bool): Whether to include a learnable bias vector.
        """
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define the learnable weight matrix W^{(l)}
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Define the optional bias vector
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        """1
        Initializes weights using Glorot uniform initialization.
        Uniform distribution in [-1/sqrt(fan_out), 1/sqrt(fan_out)].
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        """
        Forward pass of the GCN layer.
        
        Args:
            input_features: Node feature matrix H^{(l)} (sparse or dense).
            adj: Normalized adjacency matrix \tilde{A}_{norm} (sparse).
            
        Returns:
            Output feature matrix before non-linearity.
        """
        # Step 1: Linear transformation
        # Apply the trainable weight matrix: H^{(l)} W^{(l)}
        support = torch.mm(input_features, self.weight)
        
        # Step 2: Neighborhood aggregation
        # Aggregate features from neighbors: \tilde{A}_{norm} (H^{(l)} W^{(l)})
        output = torch.sparse.mm(adj, support)
        
        # Add bias if applicable
        if self.bias is not None:
            output = output + self.bias
            
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
