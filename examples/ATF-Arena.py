import torch
import torch.nn as nn
import copy
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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'double', 'ship', 'truck')

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
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

# 3. Define WideResNet-28-2 (Efficient for CIFAR)
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropout_rate=0.3, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        

# 3. Setup Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
arena_model = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0).to(device)
initial_state = copy.deepcopy(arena_model.state_dict()) 
criterion = FocalLoss(gamma=1.0)

optimizers_to_test = [
{
        "name": "HIOptimizer_Aggressive",
        "opt": HIOptimizer(arena_model.parameters(), 
                           lr=0.07, 
                           stiffening_factor=0.003, # Etwas mehr Reibung für Stabilität
                           metric_scale=200.0,      # Geometrie nutzen, nicht missbrauchen
                           beta=0.95,                # Stabilere Metrik-Glättung
                           cooling_rate=0.99,
                           epsilon=1e-7)
    }#,
    #{
    #    "name": "AdamW_Tuned",
    #    "opt": torch.optim.AdamW(arena_model.parameters(), lr=3e-3, weight_decay=0.05)
    #}
]

# --- DER BATTLE LOOP ---
for contender in optimizers_to_test:
    print(f"\n🚀 BATTLE START: {contender['name']}")
    
    # Modell auf Start zurücksetzen
    arena_model.load_state_dict(initial_state)
    optimizer = contender['opt']
    cumulative_tflops = 0.0
    
    for epoch in range(1, 5): # 4 Epochen Battle
        arena_model.train()
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = arena_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Step ausführen
            result = optimizer.step()
            
            # FLOPs Tracking (basierend auf deinen Werten)
            # 3x für Forward + Backward
            step_flops = (0.4853 * 1e9 * 3) * inputs.size(0) 
            cumulative_tflops += step_flops / 1e12
            
            if i % 100 == 99:
                status = result[0] if isinstance(result, tuple) else "Active"
                print(f"[{epoch}, {i+1}] loss: {loss.item():.3f} | status: {status}")

        # Accuracy Check nach der Epoche
        accuracy = evaluate_model(arena_model, testloader, device)
        atf = accuracy / max(1e-6, cumulative_tflops)
        
        print(f"🏁 {contender['name']} - Epoch {epoch} - Accuracy: {accuracy:.2f}%")
        print(f"📊 TFLOPs: {cumulative_tflops:.2f} | ATF: {atf:.2f}%/TFLOP")
