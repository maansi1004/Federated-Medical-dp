import torch
import torch.nn as nn

class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()
        # Layer 1: Grayscale Input -> 16 Filter Channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(2, 16) # Opacus-safe alternative to BatchNorm
        
        # Layer 2: 16 Channels -> 32 Filter Channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 32)
        
        # Spatial reduction pooling down to 14x14 grid
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classification Head: 32 channels * 14 * 14 features = 6272
        self.fc = nn.Linear(6272, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten cleanly to 6272 features
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