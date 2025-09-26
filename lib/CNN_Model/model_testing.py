import torch
import torchvision
from torchvision import datasets, transforms
from Quantization_CNN_Model import QuantizedCNN_Net

class test_model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
            )
    
    def get_accuracy(self, dequantized_dict):
        test_dataset = torchvision.datasets.MNIST(
            root='./data',        
            train=False,           
            download=True,        
            transform=self.transform  
            )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                    shuffle=False)

        quant_model = QuantizedCNN_Net().to(self.device)
        quant_model.load_quantized_tensors(dequantized_dict)
        quant_model.eval()

        with torch.no_grad():
            number_correct = 0
            samples = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = quant_model(images)

                _, prediction = torch.max(outputs, 1)
                samples += labels.size(0)
                number_correct += (prediction == labels).sum().item()

        accuracy = (number_correct / samples) * 100.0
        return accuracy