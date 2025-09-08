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
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
          )

    def forward(self, x):
        return self.model(x)
    
    def classify(self, x):
        return F.softmax(x, dim=1)


model = CNN_Net()
model_fp32 = torch.load('mnist_cnn_model.pth')
model_fp32.eval() 

# Model quantization fp32 -> int8
converter = tf.lite.TFLiteConverter.from_saved_model('cnn_fp32_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Convert weights to int8
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.float32  # Keep input in FP32
converter.inference_output_type = tf.float32  # Keep output in FP32

model_int8 = converter.convert()

# Save quantized model
with open('cnn_int8_model.tflite', 'wb') as f:
  f.write(model_int8)