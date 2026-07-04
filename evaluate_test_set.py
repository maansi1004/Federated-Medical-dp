import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import SimpleMedicalCNN

def check_clinical_metrics(thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    model = SimpleMedicalCNN()
    
    try:
        # Load your original 86% federated checkpoint model file
        state_dict = torch.load("federated_model.pth")
        clean_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)
        print("💾 Success! Loaded original custom Federated CNN weights cleanly.")
    except Exception as e:
        print(f"[ERROR] Could not load weights: {e}")
        print("Hint: If keys mismatch, delete 'federated_model.pth' and re-run your training pipeline.")
        return
        
    model.eval()

    # Load dataset
    raw_data = np.load('pneumoniamnist.npz')
    X_test = raw_data['test_images']
    y_test = raw_data['test_labels'].squeeze()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(pneumonia)
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    print("\n" + "="*60)
    print("🏥 CLINICALLY ALIGNED EVALUATION REPORT — THRESHOLD SWEEP")
    print("="*60)
    print(f"{'Threshold':<10}{'Accuracy':<12}{'Sensitivity':<14}{'Specificity':<12}")
    print("-"*60)

    for t in thresholds:
        predicted = (all_probs > t).long()
        TP = ((all_labels == 1) & (predicted == 1)).sum().item()
        FP = ((all_labels == 0) & (predicted == 1)).sum().item()
        TN = ((all_labels == 0) & (predicted == 0)).sum().item()
        FN = ((all_labels == 1) & (predicted == 0)).sum().item()

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"{t:<10}{accuracy*100:<11.2f}%{sensitivity*100:<13.2f}%{specificity*100:<11.2f}%")

    print("="*60)

if __name__ == "__main__":
    check_clinical_metrics()