# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:
Elamaran S E
### Register Number:
212222230036
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*X+1+e

plt.scatter(X.numpy(), y.numpy(), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Data for Linear Regression')
plt.show()

class Model(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear=nn.Linear(in_features, out_features)

  def forward(self,x):
      return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("Name:Elamaran S E")
print("Register Number: 212222230036")
print(f"Initial Weight: {initial_weight:.8f}, \nInitial Bias: {initial_bias:.8f}")

loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)

epoch=100
losses=[]

for epoch in range(1,epoch+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  print(f"\nEpoch: {epoch:2}, \nLoss: {loss.item():10.8f}, \nWeight: {model.linear.weight.item():10.8f}, \nBias: {model.linear.bias.item():10.8f}")

plt.plot(range(epoch),losses,color='purple')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs (Loss Curve)")
plt.show()

final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("Name:Elamaran S E")
print("Register Number: 212222230036")
print(f"Final Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}")


x1=torch.tensor([X.min().item(), X.max().item()])
y1=x1*final_weight+final_bias
print(x1)

print(y1)

plt.scatter(X.numpy(), y.numpy(), color='blue', label='Original Data')
plt.plot(x1.numpy(), y1.numpy(), color='pink', label='Best-Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()



x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("Name:Elamaran S E")
print("\nRegister Number: 212222230036")
print(f"\nPredicted for x = 120: {y_new_pred:.8f}")
```

### Dataset Information
![image](https://github.com/user-attachments/assets/46dd3995-ddd0-481d-8061-0ada66de9cfd)

### OUTPUT

![image](https://github.com/user-attachments/assets/9f501148-1acc-430f-9073-9ff76c43e501)

![image](https://github.com/user-attachments/assets/1f485797-b87d-450e-a507-06c1cd24a5b4)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/8cec68d3-ebf4-45d0-9044-5aa7fa1e676f)

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
