import torch
import numpy as np
import matplotlib.pyplot as plt

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state_dict = torch.load('mnist_cnn_model.pth', map_location=device)

# Get all parameters
params = list(model_state_dict.items())
n_params = len(params)

# Create subplots
fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8))
axes = axes.flatten() if n_params > 1 else [axes]

# Plot each parameter
for i, (name, tensor) in enumerate(params):
    values = tensor.cpu().numpy().flatten()
    axes[i].hist(values, bins=50)
    axes[i].set_title(name)
    axes[i].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()
