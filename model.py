import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2) 

    def forward(self, x):
        x = x.float() 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- ADD THIS FUNCTION AT THE BOTTOM ---
def evaluate_model(model, data_loader):
    """Utility function to test model accuracy on any dataset slice"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(data_loader), correct / total