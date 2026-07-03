import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from model import SimpleMedicalCNN
from data_setup import get_hospital_data

def train_baseline():
    # 1. Grab the loaders for all three distinct hospital nodes
    loader_A, loader_B, loader_C = get_hospital_data()
    
    # 2. Extract the underlying datasets and pool them together completely
    # This simulates a centralized, non-private world where all data is shared.
    unified_dataset = ConcatDataset([
        loader_A.dataset, 
        loader_B.dataset, 
        loader_C.dataset
    ])
    
    # Create a single master DataLoader for baseline training
    centralized_loader = DataLoader(unified_dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize the vanilla control model structure
    model = SimpleMedicalCNN()
    
    # Standard textbook loss function. We do not need heavy class weights anymore
    # because pooling the three hospitals naturally balances out the regional skews!
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🚀 Starting Centralized Baseline Training...")
    print(f"📊 Combined Dataset Size: {len(unified_dataset)} samples pooled across all nodes.")
    
    # Train for 5 epochs to ensure stable convergence
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        
        for images, labels in centralized_loader:
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
            
        epoch_loss = running_loss / len(centralized_loader)
        epoch_acc = (correct / total) * 100
        
        zeros_predicted = all_preds.count(0)
        ones_predicted = all_preds.count(1)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"    [Diagnostic] Predicted Normal (0): {zeros_predicted}, Predicted Pneumonia (1): {ones_predicted}")

    # 4. Save this control state distinctly for evaluation side-by-sides
    torch.save(model.state_dict(), "centralized_baseline.pth")
    print("\n💾 Success! 'centralized_baseline.pth' saved successfully.")

if __name__ == "__main__":
    train_baseline()