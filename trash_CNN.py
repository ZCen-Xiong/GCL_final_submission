#37257015 XIONGZICEN 20250712
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='Train CNN for waste classification')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
args = parser.parse_args()
num_epochs = args.epochs

start_time = time.time()
print(f"Starting CNN training with {num_epochs} epochs...")

dataset = load_dataset('rootstrap-org/waste-classifier', split='train')
print(dataset) 

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),  
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = dataset.shuffle(seed=42)
train_size = int(0.8 * len(dataset))    # Train/test split: 80%/20%
test_size = len(dataset) - train_size
train_dataset = dataset.select(range(train_size))
test_dataset  = dataset.select(range(train_size, len(dataset)))

X_train = []
y_train = []
for example in train_dataset:
    img = example['image']
    label = example['label']
    img_t = transform(img)
    X_train.append(img_t)
    y_train.append(label)
X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = []
y_test = []
for example in test_dataset:
    img = example['image']
    label = example['label']
    img_t = transform(img)
    X_test.append(img_t)
    y_test.append(label)
X_test = torch.stack(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)    # Batch size: 32
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):    # Classes: 7
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)    # Conv1: 3->32 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # Conv2: 32->64 channels  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Conv3: 64->128 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                   # MaxPool: 2x2
        self.fc1 = nn.Linear(128 * 8 * 8, 128)                                             # FC1: 8192->128
        self.fc2 = nn.Linear(128, num_classes)                                             # FC2: 128->7
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = SimpleCNN(num_classes=7)
cnn_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)    # Learning rate: 1e-3

cnn_model.train()
training_start_time = time.time()
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"[CNN] Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

training_time = time.time() - training_start_time
print(f"Training completed in {training_time:.2f} seconds")

cnn_model.eval()
y_pred_probs = []
y_true_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        probs = F.softmax(outputs, dim=1)
        y_pred_probs.append(probs)
        y_true_labels.append(labels)
y_pred_probs = torch.cat(y_pred_probs, dim=0)
y_true = torch.cat(y_true_labels, dim=0)

_, y_pred = torch.max(y_pred_probs, dim=1)

os.makedirs("cnn_results", exist_ok=True)
thresholds = np.linspace(0, 1, 101)
precisions = []
recalls = []
f1_scores = []

y_true_np = y_true.cpu().numpy() 
for t in thresholds:
    y_pred_at_t = -1 * np.ones_like(y_true_np) 
    max_probs, max_labels = torch.max(y_pred_probs, dim=1)
    max_probs = max_probs.cpu().numpy(); max_labels = max_labels.cpu().numpy()
    mask = max_probs >= t
    y_pred_at_t[mask] = max_labels[mask]  
    tp = np.sum((y_pred_at_t == y_true_np) & (y_pred_at_t != -1))
    fp = np.sum((y_pred_at_t != -1) & (y_pred_at_t != y_true_np))
    fn = np.sum(y_pred_at_t == -1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precisions.append(precision)
    recalls.append(recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f1_scores.append(f1)

# Plot Precision-Confidence curve
plt.figure()
plt.plot(thresholds, precisions, label="Precision")
plt.title("Precision-Confidence")
plt.xlabel("Confidence Threshold")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("cnn_results/precision-confidence.png")
plt.close()

# Plot Recall-Confidence curve
plt.figure()
plt.plot(thresholds, recalls, color='orange', label="Recall")
plt.title("Recall-Confidence")
plt.xlabel("Confidence Threshold")
plt.ylabel("Recall")
plt.grid(True)
plt.savefig("cnn_results/recall-confidence.png")
plt.close()

# Plot F1-Confidence curve
plt.figure()
plt.plot(thresholds, f1_scores, color='green', label="F1-score")
plt.title("F1-Confidence")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1 Score")
plt.grid(True)
plt.savefig("cnn_results/F1-confidence.png")
plt.close()

# Plot Precision-Recall curve (Precision vs Recall)
plt.figure()
plt.plot(recalls, precisions, color='purple')
plt.title("Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("cnn_results/precision-recall.png")
plt.close()

# Compute confusion matrix at the default threshold (t=0.5 or simply using the argmax predictions)
conf_mat = confusion_matrix(y_true.cpu(), y_pred.cpu())  # Move to CPU first
classes = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]
plt.figure(figsize=(6,6))
plt.imshow(conf_mat, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45)
plt.yticks(ticks=np.arange(len(classes)), labels=classes)
plt.colorbar()
plt.tight_layout()
plt.savefig("cnn_results/confusion_matrix.png")
plt.close()

# Print total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Training time: {training_time:.2f} seconds")
print(f"Data loading and evaluation time: {total_time - training_time:.2f} seconds")
