import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from hysterograd import HIOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Setup Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Feature Extractor (Edges/Shapes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Classifier (Logic)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

# 3. Layer-Specific Optimizer Setup
# Updated for v4: grad_clip=20.0, hysteresis_scale=5.0
# Features: stiff=0.05, metric_scale=100 
# Classifier: stiff=0.01, metric_scale=10
optimizer = HIOptimizer([
    {'params': model.features.parameters(), 'stiffening_factor': 0.05, 'lr': 0.001, 'metric_scale': 100.0},
    {'params': model.classifier.parameters(), 'stiffening_factor': 0.01, 'lr': 0.001, 'metric_scale': 10.0}
], cooling_rate=0.95, initial_temp=1.0, adaptive_threshold=True, grad_clip=20.0, hysteresis_scale=5.0) 

print("Optimizer initialized with Grad Clipping (20.0) & Hysteresis Scale (5.0).")

# 4. Training Loop
epochs = 20
logs = []
prev_accuracy = 0.0
stagnation_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_losses = []
    statuses = []
    norms = []
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Standard PyTorch Step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # HIO Step
        status, g_norm, h_width = optimizer.step()
        
        running_loss += loss.item()
        epoch_losses.append(loss.item())
        statuses.append(1 if status == "Liquid" else 0)
        norms.append(g_norm)
        
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f} | status: {status} | avg_h_width: {h_width:.4f} | norm: {g_norm:.4f}')
            running_loss = 0.0

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

    # Feedback Loop (Geometric Shock)
    if accuracy - prev_accuracy < 0.5:
        stagnation_counter += 1
    else:
        stagnation_counter = 0
        
    if stagnation_counter >= 2:
        print(f"Stagnation detected (Acc: {accuracy:.2f}%, Prev: {prev_accuracy:.2f}%). Applying Shock.")
        optimizer.shock(factor=0.3)
        stagnation_counter = 0
        
    prev_accuracy = accuracy

    logs.append({
        'epoch': epoch + 1,
        'loss': np.mean(epoch_losses),
        'activity': np.mean(statuses),
        'grad_norm': np.mean(norms),
        'accuracy': accuracy
    })

print('Finished Training')

# 5. Visualization
df = pd.DataFrame(logs)
plt.figure(figsize=(12, 10))

# Plot 1: Accuracy
plt.subplot(3, 1, 1)
plt.plot(df['epoch'], df['accuracy'], label='Test Accuracy (%)', color='blue', marker='o')
plt.ylabel('Accuracy (%)')
plt.title('HIOptimizer (v4 - Clipping & Scaling) on CIFAR-10')
plt.legend(loc='upper left')

# Plot 2: Gradient Norm
plt.subplot(3, 1, 2)
plt.plot(df['epoch'], df['grad_norm'], label='Avg Gradient Norm', color='black')
plt.axhline(y=20.0, color='r', linestyle='--', label='Clip Threshold')
plt.ylabel('Norm')
plt.legend()

# Plot 3: Activity
plt.subplot(3, 1, 3)
plt.fill_between(df['epoch'], 0, df['activity'], color='green', alpha=0.3, label='Liquid Ratio')
plt.xlabel('Epochs')
plt.ylabel('Activity Ratio')
plt.legend()

plt.tight_layout()
plt.savefig('cifar_results_v4.png')
print("Results saved to cifar_results_v4.png")