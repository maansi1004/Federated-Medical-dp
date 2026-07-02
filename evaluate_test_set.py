import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import SimpleMedicalCNN

def check_clinical_metrics():
    model = SimpleMedicalCNN()
    try:
        state_dict = torch.load("federated_model.pth")
        clean_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    except FileNotFoundError:
        print("[ERROR] 'federated_model.pth' not found. Run training first!")
        return
    model.eval()

    raw_data = np.load('pneumoniamnist.npz')
    X_test = raw_data['test_images']
    y_test = raw_data['test_labels'].squeeze()

    # --- MATCHED TRADING SCALE: [0.0, 1.0] EXACTLY ---
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

    # Confusion Matrix Metrics
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for label, pred in zip(labels, predicted):
                if label == 1 and pred == 1: TP += 1
                elif label == 0 and pred == 1: FP += 1
                elif label == 0 and pred == 0: TN += 1
                elif label == 1 and pred == 0: FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print("\n" + "="*45)
    print("🏥 CLINICALLY ALIGNED EVALUATION REPORT")
    print("="*45)
    print(f"Overall Accuracy          : {accuracy*100:.2f}%")
    print(f"Sensitivity (Recall)      : {sensitivity*100:.2f}%  <-- (Rate of catching real Pneumonia)")
    print(f"Specificity               : {specificity*100:.2f}%  <-- (Rate of correctly identifying Healthy)")
    print("-"*45)
    print(f"False Negatives (Misses!) : {FN}  <-- Clinically Dangerous!")
    print(f"False Positives (Alarms)  : {FP}  <-- Caused by Class Weights/DP Noise")
    print("="*45)

if __name__ == "__main__":
    check_clinical_metrics()