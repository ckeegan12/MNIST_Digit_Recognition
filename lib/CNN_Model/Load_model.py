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
model_fp32.load_state_dict(model_state_dict, strict=False)

# Print layer names
for key in model_state_dict.keys():
  print(key)

model_fp32.eval()

print("Model Load Sucessful")


