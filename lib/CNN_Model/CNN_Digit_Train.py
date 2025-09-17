import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Hyper parameters
epoch_losses = []
train_loss_list = []
epochs = 50
lr = 0.01
batchsize = 64

# Model device assignment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Load_data:
    ### Load data frames
    def data_sets(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = torchvision.datasets.MNIST(
                root='./data',        
                train=True,           
                download=True,        
                transform=transform  
        )

        test_dataset = torchvision.datasets.MNIST(
                root='./data',        
                train=False,           
                download=True,        
                transform=transform
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                                shuffle=False)
        return train_loader, test_loader
    
train_loader, test_loader = Load_data.data_sets()

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
    
model = CNN_Net().to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

size = len(train_loader)
print(f"Start Training for {epochs} epochs")

for epoch in range(epochs):
    print(f"training cycle: {epoch}")
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        # input shape: [28, 28]

        images = images.to(device)
        labels = labels.to(device)

        # foward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # back pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        if (i % 100) == 0:
            print(f'Epoch {epoch}, Step {i+1}, Loss: {loss.item():.4f}')

    # Calculate average for this epoch only
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_loss_list.append(epoch_loss)  # One value per epoch
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss:.4f}')

print("Training Finished")

model.eval()
with torch.no_grad():
    number_correct = 0
    samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, prediction = torch.max(outputs, 1)
        samples += labels.size(0)
        number_correct += (prediction == labels).sum().item()

accuracy = (number_correct / samples) * 100.0
print(f"Accuracy of model: {accuracy:.2f}%")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_loss_list)), train_loss_list)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Save the model
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("Model saved as 'mnist_cnn_model.pth'")
