# MNIST Neural Network Digit Recognizer 
This is a deep-learning neural network. It uses the ReLu activation and SoftMax function to predict handwritten digits. It is a three layer model with one hidden layer with 128 nodes. It is able to predict a hand written digit by propagating the input pixel data with a series of weigths and bias terms that are later updated in gradient decent.

<img align="center" width="655" height="325" alt="MNIST_dataset_example" src="https://github.com/user-attachments/assets/79a0a308-f300-424d-942e-e2cbda2a9446" />

## [Example Diagram](https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06)
<img align="center" width="1400" height="663" alt="image" src="https://github.com/user-attachments/assets/e1cd6ca9-af81-46fe-86f6-7e7396e3094b" />


## Forward Propagation

1. **Calculate ($Z^{[1]}$)**: $Z^{[1]} = XW^{[1]} + b^{[1]T}$

2. **Activation ($A^{[1]}$)**: $A^{[1]} = g_{\text{ReLU}}(Z^{[1]})$

3. **Calculate ($Z^{[2]}$)**: $Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]T}$

4. **Activation ($A^{[2]}$)**: $A^{[2]} = g_{\text{softmax}}(Z^{[2]})$

## Backward Propagation

1. **Calculate ($dZ^{[2]}$)**: $dZ^{[2]} = Y - A^{[2]T}$

2. **Calculate ($dW^{[2]}$)**: $dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})$

3. **Calculate ($dB^{[2]}$)**: $dB^{[2]} = \frac{1}{m} \sum dZ^{[2]}$

4. **Calculate ($dZ^{[1]}$)**: $dZ^{[1]} = (dZ^{[2]}W^{[2]}) \odot g'^{[1]}(Z^{[1]})$

5. **Calculate ($dW^{[1]}$)**: $dW^{[1]} = \frac{1}{m} X^TdZ^{[1]}$

6. **Calculate ($dB^{[1]}$)**: $dB^{[1]} = \frac{1}{m} \sum dZ^{[1]}$

## Parameter Updates

Update the weights and biases using the following equations:

1. **Update ($W^{[2]}$)**: $W^{[2]} := W^{[2]} - \alpha dW^{[2]}$

2. **Update ($b^{[2]}$)**: $b^{[2]} := b^{[2]} - \alpha dB^{[2]T}$

3. **Update ($W^{[1]}$)**: $W^{[1]} := W^{[1]} - \alpha dW^{[1]}$

4. **Update ($b^{[1]}$)**: $b^{[1]} := b^{[1]} - \alpha dB^{[1]T}$

## Variable Shapes

m = Batch size

### Forward Propagation

- **Input ($A^{[0]}$)**:
  - Shape: $784 \times m$
  
- **($Z^{[1]} / A^{[1]}$)**:
  - Shape: $10 \times m$
  
- **Weight ($W^{[1]}$)**:
  - Shape: $784 \times hiddenunits$ (as $A^{[0]}W^{[1]} \sim Z^{[1]}$)
  
- **Bias ($b^{[1]}$)**:
  - Shape: $hiddenunits \times 1$
  
- **$(Z^{[2]} / A^{[2]}$)**:
  - Shape: $10 \times m$
  
- **Weight ($W^{[2]}$)**:
  - Shape: $10 \times hiddenunits$ (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
  
- **Bias ($b^{[2]}$)**:
  - Shape: $10 \times 1$

### Backward Propagation

- **($dZ^{[2]}$)**:
  - Shape: $m \times 10$ (from $(A^{[2]}$))
  
- **($dW^{[2]}$)**:
  - Shape: $10 \times hiddenunits$
  
- **($dB^{[2]}$)**:
  - Shape: $10 \times 1$
  
- **($dZ^{[1]}$)**:
  - Shape: $m \times hiddenunits$ (from $(A^{[1]}$))
  
- **($dW^{[1]}$)**:
  - Shape: $784 \times hiddenunits$
  
- **($dB^{[1]}$)**:
  - Shape: $hiddenunits \times 1$
 
## Parameters
learning_rate = 0.1  
epoch = 500  
hidden_units = 128  
    
# Results
<img width="330" height="575" alt="image" src="https://github.com/user-attachments/assets/da33be50-7da1-4873-a0ea-49dab0c3ed7d" />
<img width="329" height="575" alt="image" src="https://github.com/user-attachments/assets/7a0f5902-bb65-413d-a58c-b033e7b09ba0" />
<img width="317" height="575" alt="image" src="https://github.com/user-attachments/assets/f47959d9-92b4-45b3-b26b-2bd1c68bf409" />
