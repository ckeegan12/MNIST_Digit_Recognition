import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

class QuantizedCNN_Net(nn.Module):
    def __init__(self):
        super(QuantizedCNN_Net, self).__init__()
        # Initialize layers with dummy parameters (will be replaced)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(9 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        # Store quantized tensors
        self.quantized_tensors = {}
        
    def load_quantized_tensors(self, quantized_tensors_dict):
        self.quantized_tensors = quantized_tensors_dict
        
        # Load tensors into model parameters
        with torch.no_grad():
            if 'model.0.weight' in quantized_tensors_dict:
                self.conv1.weight.data = quantized_tensors_dict['model.0.weight'].float()
            if 'model.0.bias' in quantized_tensors_dict:
                self.conv1.bias.data = quantized_tensors_dict['model.0.bias'].float()
            if 'model.3.weight' in quantized_tensors_dict:
                self.conv2.weight.data = quantized_tensors_dict['model.3.weight'].float()
            if 'model.3.bias' in quantized_tensors_dict:
                self.conv2.bias.data = quantized_tensors_dict['model.3.bias'].float()
                
            if 'model.6.weight' in quantized_tensors_dict:
                self.conv3.weight.data = quantized_tensors_dict['model.6.weight'].float()
            if 'model.6.bias' in quantized_tensors_dict:
                self.conv3.bias.data = quantized_tensors_dict['model.6.bias'].float()
                
            if 'model.10.weight' in quantized_tensors_dict:
                self.fc1.weight.data = quantized_tensors_dict['model.10.weight'].float()
            if 'model.10.bias' in quantized_tensors_dict:
                self.fc1.bias.data = quantized_tensors_dict['model.10.bias'].float()
                
            if 'model.12.weight' in quantized_tensors_dict:
                self.fc2.weight.data = quantized_tensors_dict['model.12.weight'].float()
            if 'model.12.bias' in quantized_tensors_dict:
                self.fc2.bias.data = quantized_tensors_dict['model.12.bias'].float()
                
            if 'model.14.weight' in quantized_tensors_dict:
                self.fc3.weight.data = quantized_tensors_dict['model.14.weight'].float()
            if 'model.14.bias' in quantized_tensors_dict:
                self.fc3.bias.data = quantized_tensors_dict['model.14.bias'].float()

    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # FC layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def classify(self, x):
        return F.softmax(x, dim=1)