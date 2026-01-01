import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from hysterograd import HIOptimizer

# 1. Hardware & Parallelisierung
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.mkldnn.enabled = True
device = torch.device("cpu") # Wir bleiben explizit auf CPU

# 2. Modell laden & Patchen (VOR der Optimizer-Initialisierung)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# PATCH für 32x32 Bilder
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10) # Direkt auf 10 Klassen setzen

# 3. Layer einfrieren / auftauen
for param in model.parameters():
    param.requires_grad = False

# Wir trainieren Layer 4 und die neue FC-Schicht
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
# WICHTIG: Auch conv1 muss lernen, da wir das Design geändert haben!
for param in model.conv1.parameters():
    param.requires_grad = True

model.to(device)

# 4. Daten (32x32, kein Resize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Batch 64 ist effizienter für 8 Kerne

# 5. EINZIGER Optimizer-Aufruf mit aggressiveren "Break-out" Settings
optimizer = HIOptimizer(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.002,             # Höhere LR für CPU-Sprint
    stiffening_factor=0.01,
    metric_scale=10.0,    # Moderat verstärken
    hysteresis_scale=0.7, # Flüssiger
    initial_temp=1.5      # "Heißer" Start gegen das 2.4 Plateau
)

criterion = nn.CrossEntropyLoss()

# 6. Training
model.train()
print("Starte HIO-Refinement (CPU-Optimized)...")
for epoch in range(1):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            # Hier kannst du im Log sehen, ob der Loss unter 2.3 sinkt
            print(f'[{epoch + 1}, {i + 1:5d}] Loss: {running_loss / 10:.4f}')
            running_loss = 0.0
