import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from hysterograd import HIOptimizer

# 2. Setup Validierungs-Szenario
# Datensatz: Schwingende "Moons" (Nicht-lineare Klassifikation)
X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modell: Einfaches MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
optimizer = HIOptimizer(model.parameters(), lr=0.01, stiffening_factor=0.05)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop
epochs = 100
logs = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    statuses = []
    norms = []
    
    for bx, by in loader:
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        
        status, g_norm, h_width = optimizer.step()
        
        epoch_loss += loss.item()
        statuses.append(1 if status == "Liquid" else 0)
        norms.append(g_norm)
        
    avg_loss = epoch_loss / len(loader)
    model.eval()
    logs.append({
        'epoch': epoch,
        'loss': avg_loss,
        # 'tau': optimizer.tau, # Removed as it's internal per-group now
        'hysterese': h_width,   # Use the returned value
        'activity': np.mean(statuses),
        'grad_norm': np.mean(norms)
    })

# 4. Visualisierung
import pandas as pd
df = pd.DataFrame(logs)

plt.figure(figsize=(12, 8))

# Plot 1: Loss & Hysterese
plt.subplot(2, 1, 1)
plt.plot(df['epoch'], df['loss'], label='Loss', color='blue', lw=2)
plt.ylabel('Loss')
plt.title('HIOptimizer: Validierung an "Moons" Datensatz')
plt.legend(loc='upper left')
ax2 = plt.gca().twinx()
ax2.plot(df['epoch'], df['hysterese'], label='Hysterese-Breite ($E_{reset}$)', color='red', linestyle='--')
ax2.set_ylabel('Barriere-Energie')
ax2.legend(loc='upper right')

# Plot 2: Aktivität (Liquid vs Frozen)
plt.subplot(2, 1, 2)
plt.fill_between(df['epoch'], 0, df['activity'], color='green', alpha=0.3, label='Optimierungs-Aktivität (Liquid Ratio)')
plt.plot(df['epoch'], df['grad_norm'], label='Gradient Norm', color='black', alpha=0.6)
plt.xlabel('Epochen')
plt.ylabel('Intensität / Anteil')
plt.legend()

plt.tight_layout()
plt.savefig('hio_validation.png')

# Output Data
df.to_csv('validation_results.csv', index=False)
print(df.tail(10))
