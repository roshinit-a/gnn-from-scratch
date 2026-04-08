import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import GCN

def set_seed(seed: int = 42) -> None:
    """Pin all random-number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic cuDNN ops (small perf cost, worth it for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fixed hyperparameters (not exposed as CLI args)
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 16
DROPOUT = 0.5
CHECKPOINT_DIR = "../results"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_gcn_model.pth")

def accuracy(logits, labels):
    """
    Computes the accuracy of the predictions.
    
    Args:
        logits: Raw scores from the model.
        labels: Ground truth labels.
        
    Returns:
        float: Accuracy score between 0 and 1.
    """
    predictions = torch.argmax(logits, dim=1)
    correct = torch.sum(predictions == labels).item()
    return correct / len(labels)

def main():
    """
    Main training script.

    Supports CLI arguments:
        --epochs  Number of training epochs (default: 200)
        --lr      Learning rate for the Adam optimizer (default: 0.01)
        --seed    Random seed for reproducibility (default: 42)
    """
    parser = argparse.ArgumentParser(description="Train a 2-layer GCN on the Cora dataset.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam optimizer (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Fix all random seeds before anything stochastic happens
    set_seed(args.seed)

    # Ensure results directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Load Data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(data_dir="../data/cora/")
    
    # Extract dimensions
    n_features = features.shape[1]
    n_classes = int(labels.max().item() + 1)
    
    print(f"Dataset loaded. Features: {n_features}, Classes: {n_classes}")
    
    # 2. Initialize Model, Optimizer, and Loss Function
    model = GCN(n_features=n_features, 
                n_hidden=HIDDEN_DIM, 
                n_classes=n_classes, 
                dropout_rate=DROPOUT)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    # 3. Training Loop
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features, adj)
        
        # Compute loss on training nodes only
        loss_train = criterion(logits[idx_train], labels[idx_train])
        acc_train = accuracy(logits[idx_train], labels[idx_train])
        
        # Backward pass and optimization
        loss_train.backward()
        optimizer.step()
        
        # Evaluate on Validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            loss_val = criterion(val_logits[idx_val], labels[idx_val])
            acc_val = accuracy(val_logits[idx_val], labels[idx_val])
            
        # Save best model
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {loss_train.item():.4f} | "
                  f"Val Loss: {loss_val.item():.4f} | "
                  f"Val Acc: {acc_val:.4f}")
                  
    print(f"Training completed in {time.time() - start_time:.2f}s")
    
    # 4. Testing
    print("Loading the best model for testing...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    
    with torch.no_grad():
        test_logits = model(features, adj)
        test_loss = criterion(test_logits[idx_test], labels[idx_test])
        test_acc = accuracy(test_logits[idx_test], labels[idx_test])
        
    print(f"Final Test Results - Loss: {test_loss.item():.4f}, Accuracy: {test_acc:.4f} (~{test_acc*100:.1f}%)")

if __name__ == "__main__":
    main()
