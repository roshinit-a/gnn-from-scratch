import os
import urllib.request
import numpy as np
import scipy.sparse as sp
import torch

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cora")

def download_cora(data_dir=DEFAULT_DATA_DIR):
    """
    Downloads the Cora dataset (cora.content and cora.cites) if not present.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/"
    
    for filename in ["cora.content", "cora.cites"]:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)

def encode_onehot(labels):
    """
    Encodes string labels into integer indices.
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return np.where(labels_onehot)[1]

def normalize_features(features):
    """
    Row-normalize feature matrix. 
    Each node's feature vector sum is 1.
    """
    rowsum = np.array(features.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adjacency(adj):
    """
    Symmetrically normalize adjacency matrix.
    Computes D^{-1/2} A D^{-1/2}.
    """
    # 1. Compute node degrees \tilde{D} for the graph with self-loops
    rowsum = np.array(adj.sum(1))
    
    # 2. Compute D^{-1/2}
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # Handle division by zero for isolated nodes
    
    # 3. Create a diagonal matrix for D^{-1/2}
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # 4. Multiply D^{-1/2} A D^{-1/2}
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
def load_data(data_dir=DEFAULT_DATA_DIR):
    """
    Loads Cora data, preprocesses it, and returns normalized A, X, and Y along with train/val/test splits.
    """
    download_cora(data_dir)
    
    print('Loading cora dataset...')
    
    # cora.content contains: <node_id> <feature_1> ... <feature_1433> <label>
    idx_features_labels = np.genfromtxt(os.path.join(data_dir, "cora.content"), dtype=np.dtype(str))
    
    # Extract features (columns 1 to -1) and convert to sparse matrix
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    
    # Extract labels and convert to numerical indices
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # Build graph adjacency matrix
    # Extract node IDs and map them to row/col indices (0 to 2707)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # cora.cites contains: <cited_node> <citing_node> (edges are directed in file, need to make undirected)
    edges_unordered = np.genfromtxt(os.path.join(data_dir, "cora.cites"), dtype=np.int32)
    
    # Map raw node IDs to 0-based indices
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    
    # Construct Sparse Adjacency Matrix (COO format)
    # A_ij = 1 if there is an edge between node i and node j
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # Make adjacency matrix symmetric: A = A + A^T
    # We remove element-wise max duplication mathematically
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Feature normalization
    features = normalize_features(features)
    
    # Full Normalization of Adjacency Matrix (The Math)
    # Step 1: Add self connections \tilde{A} = A + I
    # We add the identity matrix so every node aggregates its own features
    adj = adj + sp.eye(adj.shape[0])
    
    # Step 2: Symmetrically normalize \tilde{A} by node degrees 
    # \tilde{A}_{norm} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
    adj = normalize_adjacency(adj)
    
    # Define standard splits: train (140), val (300), test (1000)
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    
    # Convert numpy arrays & scipy sparse matrices to PyTorch tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(list(idx_train))
    idx_val = torch.LongTensor(list(idx_val))
    idx_test = torch.LongTensor(list(idx_test))
    
    # Convert sparse scipy adjacency matrix to sparse PyTorch tensor
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    shape = torch.Size(adj.shape)
    adj = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    return adj, features, labels, idx_train, idx_val, idx_test

if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print("Features shape:", features.shape)
    print("Adjacency shape:", adj.shape)
    print("Labels shape:", labels.shape)
