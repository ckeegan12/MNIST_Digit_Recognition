import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__() 
        self.model = nn.Sequential(
          # Convolution layers
            nn.Conv2d(in_channels=1, out_channels=3,
                            kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=3, out_channels=9,
                            kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=9, out_channels=9,
                            kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(9 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
          )

    def forward(self, x):
        return self.model(x)
    
    def classify(self, x):
        return F.softmax(x, dim=1)

model_fp32 = CNN_Net()
model_state_dict = torch.load('mnist_cnn_model.pth', map_location='cpu')
model_fp32.load_state_dict(model_state_dict)

# Print layer names
for key in model_state_dict.keys():
  print(key)

# Load models weights and bias tensors from state dictionary
weight_0 = model_state_dict['model.0.weight']
weight_3 = model_state_dict['model.3.weight']
weight_6 = model_state_dict['model.6.weight']
weight_10 = model_state_dict['model.10.weight']
weight_13 = model_state_dict['model.13.weight']
weight_16 = model_state_dict['model.16.weight']

bias_0 = model_state_dict['model.0.bias']
bias_3 = model_state_dict['model.3.bias']
bias_6 = model_state_dict['model.6.bias']
bias_10 = model_state_dict['model.10.bias']
bias_13 = model_state_dict['model.13.bias']
bias_16 = model_state_dict['model.16.bias']

model_fp32.eval()

print("Model Load Sucessful")

print(f"{weight_0}\n{weight_3}\n{weight_6}\n{weight_10}\n{weight_13}\n{weight_16}\n")
print("all weight tensors shown")
print(f"{bias_0}\n{bias_3}\n{bias_6}\n{bias_10}\n{bias_13}\n{bias_16}\n")
print("all bias tensors shown")
