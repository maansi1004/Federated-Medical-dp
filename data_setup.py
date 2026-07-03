import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def oversample_minority(images, labels, target_ratio=0.4):
    """
    Duplicates minority-class samples so the model sees them more often.
    Unlike loss weighting, this survives Opacus's per-sample gradient clipping,
    since it changes how often an example is seen, not how large its gradient is.
    """
    labels_np = labels.numpy()
    class_counts = np.bincount(labels_np, minlength=2)

    # Identify which class is the minority *for this specific hospital*
    minority_class = int(np.argmin(class_counts))
    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())

    if minority_count == 0:
        # No samples of one class at all — oversampling can't help, flag it
        print(f"⚠️  Warning: one class has zero samples in this split (counts={class_counts}). Skipping oversampling.")
        return images, labels

    minority_mask = labels_np == minority_class
    minority_images = images[minority_mask]
    minority_labels = labels[minority_mask]

    # How many total minority samples we want, given the target ratio
    target_minority_count = int(target_ratio * majority_count / (1 - target_ratio))
    repeats =  target_minority_count // minority_count
    if repeats < 1:
    # Minority class already meets or exceeds the target ratio — leave as-is
        return images, labels
    images_oversampled = torch.cat([images] + [minority_images] * repeats, dim=0)
    labels_oversampled = torch.cat([labels] + [minority_labels] * repeats, dim=0)

    return images_oversampled, labels_oversampled


def get_hospital_data(batch_size=32, seed=42, oversample_target_ratio=0.4):
    np.random.seed(seed)

    raw_data = np.load('pneumoniamnist.npz')
    X_train = raw_data['train_images']
    y_train = raw_data['train_labels'].squeeze()

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)

    normal_idx = np.where(y_train.numpy() == 0)[0]
    pneumonia_idx = np.where(y_train.numpy() == 1)[0]

    np.random.shuffle(normal_idx)
    np.random.shuffle(pneumonia_idx)

    # Precise 80% Label Skew Distribution
    p_split1 = int(0.80 * len(pneumonia_idx))
    n_split1 = int(0.10 * len(normal_idx))

    p_split2 = p_split1 + int(0.10 * len(pneumonia_idx))
    n_split2 = n_split1 + int(0.10 * len(normal_idx))

    node_A_idx = np.concatenate([pneumonia_idx[:p_split1], normal_idx[:n_split1]])
    node_B_idx = np.concatenate([pneumonia_idx[p_split1:p_split2], normal_idx[n_split1:n_split2]])
    node_C_idx = np.concatenate([pneumonia_idx[p_split2:], normal_idx[n_split2:]])

    def build_loader(indices, hospital_name):
        images = X_train[indices]
        labels = y_train[indices]

        pre_counts = np.bincount(labels.numpy(), minlength=2)
        images, labels = oversample_minority(images, labels, target_ratio=oversample_target_ratio)
        post_counts = np.bincount(labels.numpy(), minlength=2)

        print(f"🏥 {hospital_name}: {pre_counts} -> {post_counts} (Normal, Pneumonia) after oversampling")

        return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)

    print("=== Clean Non-IID Dataset Splits (with per-hospital oversampling) ===")
    loader_A = build_loader(node_A_idx, "Hospital A (Pneumonia Hotspot)")
    loader_B = build_loader(node_B_idx, "Hospital B (Community Screening)")
    loader_C = build_loader(node_C_idx, "Hospital C (Healthy Clinic Shadow)")
    print()

    return loader_A, loader_B, loader_C