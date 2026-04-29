import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Sample Dataset class
class GaitDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# LSTM-based model for gait pattern detection
class GaitPatternDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GaitPatternDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

# Training function
def train_model_with_visualization(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(dataloader)
        epoch_accuracy = correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Plot training metrics
    plot_training_metrics(losses, accuracies)

# Visualization function for training metrics
def plot_training_metrics(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Hyperparameters
input_size = 10  # Number of features (e.g., sensor data dimensions)
hidden_size = 64
num_layers = 2
num_classes = 3  # Example: 3 gait patterns (normal, limping, etc.)
num_epochs = 20
batch_size = 32
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data (replace with real sensor data)
train_data = torch.randn(100, 50, input_size)  # 100 samples, 50 time steps, 10 features
train_labels = torch.randint(0, num_classes, (100,))

dataset = GaitDataset(train_data, train_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, and optimizer
model = GaitPatternDetector(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model_with_visualization(model, dataloader, criterion, optimizer, num_epochs)

# Save the model
torch.save(model.state_dict(), "gait_pattern_detector.pth")