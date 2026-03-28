import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SpectrumCNN, self).__init__()
        # Input shape: (Batch Size, 1, 64, 64)
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: (16, 32, 32)
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: (32, 16, 16)
        
        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # Output: (64, 8, 8)
        
        # Fully Connected
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes) 
        # Classes: 0=Idle, 1=5G, 2=IoT, 3=Radar

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class InferenceEngine:
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = SpectrumCNN().to(self.device)
        self.model.eval()
        self.class_names = ["Idle", "5G", "IoT", "Radar"]
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    @torch.no_grad()
    def predict(self, spectrogram_tensor):
        """
        Predicts if the spectrum is Idle or contains a specific signal structure.
        Returns -> bool (is_busy), str (class_name), float (confidence)
        """
        if spectrogram_tensor.dim() == 3:
            spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
            
        spectrogram_tensor = spectrogram_tensor.to(self.device).float()
        
        outputs = self.model(spectrogram_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        confidence, predicted = torch.max(probabilities, 1)
        idx = predicted.item()
        
        is_busy = bool(idx != 0)
        return is_busy, self.class_names[idx], confidence.item()
