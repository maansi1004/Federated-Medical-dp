
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_hospital_data(batch_size=32, seed=42):
    # Set seed for reproducible shuffling
    np.random.seed(seed)
    
    # Load raw data
    raw_data = np.load('pneumoniamnist.npz')
    X_train = raw_data['train_images']
    y_train = raw_data['train_labels'].squeeze()
    
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    # Separate indices by class
    normal_idx = np.where(y_train.numpy() == 0)[0]
    pneumonia_idx = np.where(y_train.numpy() == 1)[0]
    
    # --- FIXED: Shuffle within classes to remove acquisition order artifacts ---
    np.random.shuffle(normal_idx)
    np.random.shuffle(pneumonia_idx)
    
    # --- FIXED: Precise 80% Label Skew Distribution ---
    # Hospital A (Pneumonia Hotspot): 80% of all Pneumonia, 10% of all Normal
    p_split1 = int(0.80 * len(pneumonia_idx))
    n_split1 = int(0.10 * len(normal_idx))
    
    # Hospital B (Balanced Leftovers): 10% of all Pneumonia, 10% of all Normal
    p_split2 = p_split1 + int(0.10 * len(pneumonia_idx))
    n_split2 = n_split1 + int(0.10 * len(normal_idx))
    
    # Hospital C (Healthy Clinic): Remaining 10% of Pneumonia, 80% of Normal
    
    node_A_idx = np.concatenate([pneumonia_idx[:p_split1], normal_idx[:n_split1]])
    node_B_idx = np.concatenate([pneumonia_idx[p_split1:p_split2], normal_idx[n_split1:n_split2]])
    node_C_idx = np.concatenate([pneumonia_idx[p_split2:], normal_idx[n_split2:]])
    
    def build_loader(indices):
        return DataLoader(TensorDataset(X_train[indices], y_train[indices]), batch_size=batch_size, shuffle=True)
    
    print("=== Clean Non-IID Dataset Splits ===")
    print(f"🏥 Hospital A (Pneumonia Hotspot): {len(node_A_idx)} samples")
    print(f"🏥 Hospital B (Community Screening): {len(node_B_idx)} samples")
    print(f"🏥 Hospital C (Healthy Clinic Shadow): {len(node_C_idx)} samples\n")
    
    return build_loader(node_A_idx), build_loader(node_B_idx), build_loader(node_C_idx)