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

# 2. Define Model (Enhanced SimpleCNN - VGG Style for High Capacity)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
                self.features = nn.Sequential(
                    conv_block(3, 64),    # 32x32 -> 16x16
                    conv_block(64, 128),  # 16x16 -> 8x8
                    conv_block(128, 256), # 8x8 -> 4x4
                    nn.MaxPool2d(4)       # 4x4 -> 1x1
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 10)
                )
        
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # FLOPs Counter
        def count_flops(model, input_size):
            flops = 0
            def conv2d_flops(m, x, y):
                nonlocal flops
                h, w = y.shape[2], y.shape[3]
                ops = 2 * m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels * h * w
                flops += ops
        
            def linear_flops(m, x, y):
                nonlocal flops
                ops = 2 * m.in_features * m.out_features
                flops += ops
        
            hooks = []
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    hooks.append(m.register_forward_hook(conv2d_flops))
                elif isinstance(m, nn.Linear):
                    hooks.append(m.register_forward_hook(linear_flops))
        
            # Dummy pass
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_size).to(device)
                model(dummy_input)
        
            for h in hooks:
                h.remove()
                
            return flops
        
        # Calculate FLOPs per image
        flops_per_img = count_flops(model, (3, 32, 32))
        print(f"FLOPs per image: {flops_per_img / 1e9:.4f} GFLOPs")
        
        # 3. Layer-Specific Optimizer Setup (Aggressive HIO Optimization)
        # Using high metric_scale to force strong natural gradient steps
        optimizer = HIOptimizer([
            {'params': model.features.parameters(), 'stiffening_factor': 0.02, 'lr': 0.08, 'metric_scale': 1000.0},
            {'params': model.classifier.parameters(), 'stiffening_factor': 0.01, 'lr': 0.1, 'metric_scale': 100.0}
        ],
        adaptive_threshold=True,
        grad_clip=2.0,
        hysteresis_scale=1.2
        )
# 4. Training Loop
epochs = 10
logs = []
prev_accuracy = 0.0
stagnation_counter = 0
total_pflops = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_losses = []
    statuses = []
    norms = []
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        batch_len = inputs.size(0)

        # Standard PyTorch Step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # HIO Step
        status, g_norm, h_width = optimizer.step()
        
        # Update PFLOPs
        # Total FLOPs = flops_per_img * batch_size * 3 (Forward + Backward approx 2x Forward)
        # Usually Backward is ~2x Forward. So Total ~ 3x Forward.
        current_flops = flops_per_img * batch_len * 3
        total_pflops += current_flops / 1e15 # Convert to Peta FLOPs
        
        running_loss += loss.item()
        epoch_losses.append(loss.item())
        
        # Parse Status for Activity Logging
        if "Liquid" in status:
            statuses.append(1.0)
        elif "Viscous" in status:
            try:
                val = float(status.split('(')[1].strip(')'))
                statuses.append(val)
            except:
                statuses.append(0.5)
        else: # Frozen
            statuses.append(0.0)

        norms.append(g_norm)
        
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f} | status: {status} | avg_h_width: {h_width:.4f} | norm: {g_norm:.4f} | PFLOPs: {total_pflops:.9f}')
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
            
            # Count inference FLOPs? Usually negligible compared to training but technically adds up.
            total_pflops += (flops_per_img * images.size(0)) / 1e15

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}% | Total PFLOPs: {total_pflops:.9f}')

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
plt.title('HIOptimizer (v5 - Inverted Geometric Coupling) on CIFAR-10')
plt.legend(loc='upper left')

# Plot 2: Gradient Norm
plt.subplot(3, 1, 2)
plt.plot(df['epoch'], df['grad_norm'], label='Avg Gradient Norm', color='black')
plt.axhline(y=20.0, color='r', linestyle='--', label='Clip Threshold')
plt.ylabel('Norm')
plt.legend()

# Plot 3: Activity
plt.subplot(3, 1, 3)
plt.fill_between(df['epoch'], 0, df['activity'], color='green', alpha=0.3, label='Liquid Ratio (Gate Avg)')
plt.xlabel('Epochs')
plt.ylabel('Activity Ratio')
plt.legend()

plt.tight_layout()
plt.savefig('cifar_results_v5.png')
print("Results saved to cifar_results_v5.png")
