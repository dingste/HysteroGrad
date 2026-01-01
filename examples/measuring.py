import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from hysterograd import HIOptimizer
import os

# Ensure docs/images exists
os.makedirs('docs/images', exist_ok=True)

# 1. Setup Data with Stronger Augmentation
print("Setting up data...")
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
#    Cutout(n_holes=1, length=8)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

batch_size = 64

# Use existing data directory
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define Focal Loss
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

# 4. Setup Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''
WRN-16-1:
Epoch 20: Hysteresis Scale = 0.70
Epoch 20/20 completed. Avg Loss: 0.2466
==============================
PERFORMANCE METRICS
==============================
Accuracy:  0.8473 (84.73%)
Precision: 0.8531
Recall:    0.8473
F1 Score:  0.8477
==============================
'''
model = WideResNet(depth=22, widen_factor=1, dropout_rate=0.0).to(device) 
criterion = FocalLoss(gamma=1.0)

optimizer = HIOptimizer([
    {'params': model.parameters(), 'stiffening_factor': 0.003, 'lr': 0.05, 'metric_scale': 300.0, 'beta': 0.9, 'epsilon': 1e-8, 'cooling_rate': 0.995 },
],
adaptive_threshold=True,
grad_clip=1.5, 
hysteresis_scale=0.6
)


'''
Epoch 1/20 completed. Avg Loss: 1.6203
Epoch 6/20 completed. Avg Loss: 0.5384
Epoch 20/20 completed. Avg Loss: 0.2182
==============================
PERFORMANCE METRICS
==============================
Accuracy:  0.8758 (87.58%)
Precision: 0.8862
Recall:    0.8758
F1 Score:  0.8769
==============================
' ''

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = HIOptimizer([
    {'params': model.features.parameters(), 'stiffening_factor': 0.02, 'lr': 0.08, 'metric_scale': 1000.0},
    {'params': model.classifier.parameters(), 'stiffening_factor': 0.01, 'lr': 0.1, 'metric_scale': 100.0}
],
adaptive_threshold=True,
grad_clip=2.0,
hysteresis_scale=0.9
)
'''

# 5. Training Loop
epochs = 40
print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # Scheduled Hysteresis Increase: 0.6 -> 0.9
    new_hyst = 0.6 + (0.3 * (epoch / max(1, epochs - 1)))
    for group in optimizer.param_groups:
        group['hysteresis_scale'] = new_hyst
    
    print(f"Epoch {epoch+1}: Hysteresis Scale = {new_hyst:.2f}")

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        #if i % 20 == 0:
        #     print(f"Epoch {epoch+1}, Batch {i}/{len(trainloader)}")
        
    print(f'Epoch {epoch + 1}/{epochs} completed. Avg Loss: {running_loss / len(trainloader):.4f}')

print('Finished Training')

# 6. Evaluation and Metrics
print("Evaluating model...")
model.eval()
all_preds = []
all_labels = []

# For Grad-CAM (Target last layer)
#target_layer = model.layer3[-1].conv2
grad_cam_maps = []
grad_cam_images = []
grad_cam_labels = []

# Hook for Grad-CAM
feature_maps = []
gradients = []

def save_feature_map(module, input, output):
    feature_maps.append(output)

def save_gradient(module, grad_in, grad_out):
    gradients.append(grad_out[0])

#handle_f = target_layer.register_forward_hook(save_feature_map)
#handle_b = target_layer.register_full_backward_hook(save_gradient)

# Only run grad cam on first batch
grad_cam_done = False


with torch.no_grad(): # Default no_grad for evaluation
    for i, data in enumerate(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        
        if not grad_cam_done and i == 0:
            # Special pass for Grad-CAM (requires grad)
            with torch.enable_grad():
                # Pick first 5 images
                gc_images = images[:5]
                gc_labels = labels[:5]
                
                outputs = model(gc_images)
                '''
                # Backward for each image to get specific class gradients
                for j in range(5):
                    feature_maps.clear()
                    gradients.clear()
                    
                    # Rerun single forward to catch correct features/grads strictly coupled
                    # (Simplified: just use batch, but we need per-sample grad)
                    # For efficiency, we just used batch output, but let's do one by one for clarity
                    single_img = gc_images[j:j+1]
                    single_lbl = gc_labels[j]
                    
                    out = model(single_img)
                    model.zero_grad()
                    score = out[0, single_lbl]
                    score.backward()
                    
                    # Compute Grad-CAM
                    # Grads: [1, C, H, W]
                    # Feats: [1, C, H, W]
                    grads = gradients[0].cpu().data.numpy()[0]
                    feats = feature_maps[0].cpu().data.numpy()[0]
                    
                    weights = np.mean(grads, axis=(1, 2))
                    cam = np.zeros(feats.shape[1:], dtype=np.float32)
                    
                    for k, w in enumerate(weights):
                        cam += w * feats[k]
                        
                    cam = np.maximum(cam, 0)
                    cam = cv2.resize(cam, (32, 32)) if 'cv2' in globals() else np.array(transforms.Resize((32,32))(torch.tensor(cam).unsqueeze(0)).squeeze(0))
                    cam = cam - np.min(cam)
                    cam = cam / (np.max(cam) + 1e-8)
                    
                    grad_cam_maps.append(cam)
                    grad_cam_images.append(single_img.cpu())
                    grad_cam_labels.append(single_lbl.item())
                    
            grad_cam_done = True
            # Clear hooks
            handle_f.remove()
            handle_b.remove()
       '''     
            # Continue with normal eval (need to re-run the batch without hooks/grads if we want strict consistency, but whatever)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            continue

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print("\n" + "="*30)
print("PERFORMANCE METRICS")
print("="*30)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("="*30)

print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 7. Visualizations

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('docs/images/confusion_matrix.png')
print("Saved Confusion Matrix to docs/images/confusion_matrix.png")

# Grad-CAM Visualization
if grad_cam_maps:
    plt.figure(figsize=(15, 5))
    for i in range(len(grad_cam_maps)):
        plt.subplot(1, 5, i+1)
        
        # Denormalize image for display
        img_tensor = grad_cam_images[i][0]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array(stats[1]) + np.array(stats[0]) # Un-normalize
        img = np.clip(img, 0, 1)
        
        cam = grad_cam_maps[i]
        
        plt.imshow(img)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f"{classes[grad_cam_labels[i]]}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('docs/images/grad_cam.png')
    print("Saved Grad-CAM to docs/images/grad_cam.png")
