import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from model import SimpleMedicalCNN
from data_setup import get_hospital_data

def train_baseline():
    train_loader_A, _, _ = get_hospital_data()
    model = SimpleMedicalCNN()
    
    # --- FIX: Penalize the model heavily for missing the rare class (0) ---
    # We give Class 0 (Normal) a weight of 4.0, and Class 1 (Pneumonia) a weight of 1.0
    class_weights = torch.tensor([4.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ----------------------------------------------------------------------
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Weighted Baseline Training on Hospital A...")
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        
        for images, labels in train_loader_A:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader_A)
        epoch_acc = (correct / total) * 100
        
        zeros_predicted = all_preds.count(0)
        ones_predicted = all_preds.count(1)
        
        print(f"Epoch {epoch+1}/3 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"    [Diagnostic] Predicted Normal (0): {zeros_predicted}, Predicted Pneumonia (1): {ones_predicted}")

if __name__ == "__main__":
    train_baseline()