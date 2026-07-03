import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # FIX: 4 groups processing 16 channels (4 channels per group)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # FIX: 4 groups processing 32 channels (8 channels per group)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=32)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- OPTIMIZATION: Global Average Pooling (GAP) ---
        # Instead of a massive flattening layer that blows up parameter counts,
        # GAP reduces the spatial dimensions (H x W) down to a 1x1 vector per channel.
        # This slashes parameter sizes, heavily reducing the vector space vulnerable to DP noise!
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Classifier Head
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # Block 1: Conv -> GroupNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        
        # Block 2: Conv -> GroupNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        
        # Compress spatial structures down efficiently
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) 
        
        # Output Logits
        return self.fc(x)

def evaluate_model(model, data_loader):
    """Utility function for local client evaluation loops."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0.0, 0, 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return loss / total, correct / total