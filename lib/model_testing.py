import torch
import torchvision
from torchvision import transforms
from Load_model import CNN_Net

batchsize = 64

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
)

test_dataset = torchvision.datasets.MNIST(
        root='./data',        
        train=False,           
        download=True,        
        transform=transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                           shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Net().to(device)
state_dict = torch.load('mnist_cnn_model.pth', map_location=device)
model.load_state_dict(state_dict)

model.eval()
number_correct = 0
samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, prediction = torch.max(outputs, 1)
        samples += labels.size(0)
        number_correct += (prediction == labels).sum().item()

accuracy = (number_correct / samples) * 100.0
print(f"Accuracy of model: {accuracy:.2f}%")
