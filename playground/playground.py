import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class ThreeLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ThreeLayerPerceptron, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.hidden.weight, gain=1.0)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.output.weight, gain=1.0)
        torch.nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

class SVDLoRALayer(nn.Module):
    def __init__(self, weight_matrix, r_svd, r_lora, seed=42):
        super(SVDLoRALayer, self).__init__()
        torch.manual_seed(seed)  # Set seed for initialization
        
        # Perform SVD
        U, S, Vt = torch.linalg.svd(weight_matrix)
        
        # Keep top r_svd singular values and vectors
        self.U = nn.Parameter(U[:, :r_svd], requires_grad=False)
        self.S = nn.Parameter(S[:r_svd], requires_grad=True)
        self.V = nn.Parameter(Vt[:r_svd, :].t(), requires_grad=False)
        
        # Initialize LoRA matrices with fixed seed
        self.X = nn.Parameter(torch.randn(r_svd, r_lora, generator=torch.Generator().manual_seed(seed)) * 0.01)
        self.Y = nn.Parameter(torch.randn(r_lora, r_svd, generator=torch.Generator().manual_seed(seed + 1)) * 0.01)
    
    def forward(self):
        # Compute S + XY
        S_modified = torch.diag(self.S) + self.X @ self.Y
        # Reconstruct weight matrix
        return self.U @ S_modified @ self.V.t()

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    training_loss = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_loader)
        training_loss.append(avg_loss)
        print(f'Epoch {epoch+1}, Average loss: {avg_loss:.4f}')
    
    return training_loss

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    return accuracy_score(targets, predictions), confusion_matrix(targets, predictions)

def analyze_performance_by_digit(confusion_matrix):
    per_digit_accuracy = {}
    for i in range(10):
        true_positives = confusion_matrix[i, i]
        total = confusion_matrix[i, :].sum()
        accuracy = true_positives / total
        per_digit_accuracy[i] = accuracy
    return per_digit_accuracy

def plot_training_curves(initial_loss, fine_tuning_loss):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(initial_loss)
    plt.title('Initial Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(fine_tuning_loss)
    plt.title('Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def main():
    SEED = 42
    set_all_seeds(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    input_size = 28 * 28
    hidden_size = 256
    num_classes = 10
    model = ThreeLayerPerceptron(input_size, hidden_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining initial model...")
    start_time = time.time()
    initial_loss = train_model(model, train_loader, criterion, optimizer, device, epochs=10)
    initial_training_time = time.time() - start_time
    
    accuracy, conf_matrix = evaluate_model(model, test_loader, device)
    print(f"\nInitial Model Accuracy: {accuracy:.4f}")
    
    digit_accuracies = analyze_performance_by_digit(conf_matrix)
    print("\nPer-digit accuracies:")
    for digit, acc in digit_accuracies.items():
        print(f"Digit {digit}: {acc:.4f}")
    
    # Find worst performing digit
    worst_digit = min(digit_accuracies.items(), key=lambda x: x[1])[0]
    print(f"\nWorst performing digit: {worst_digit}")
    
    worst_digit_indices = [i for i, (_, label) in enumerate(train_dataset)
                          if label == worst_digit]
    worst_digit_dataset = Subset(train_dataset, worst_digit_indices)
    worst_digit_loader = DataLoader(
        worst_digit_dataset, 
        batch_size=64, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Apply SVD-LoRA fine-tuning to hidden layer
    print("\nApplying SVD-LoRA fine-tuning for worst performing digit...")
    weight_matrix = model.hidden.weight.data
    print("\nThe Rank of the weight matrix:" , weight_matrix.shape)
    r_svd = min(70, min(weight_matrix.shape))
    r_lora = 10
    
    svd_lora_layer = SVDLoRALayer(weight_matrix, r_svd, r_lora, seed=SEED).to(device)
    optimizer_ft = optim.Adam([svd_lora_layer.S, svd_lora_layer.X, svd_lora_layer.Y], lr=0.0005)
    
    # Fine-tuning
    start_time = time.time()
    fine_tuning_loss = []
    
    for epoch in range(10):
        svd_lora_layer.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(worst_digit_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer_ft.zero_grad()
            
            # Update hidden layer weight matrix
            model.hidden.weight.data = svd_lora_layer()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_ft.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(worst_digit_loader)
        fine_tuning_loss.append(avg_loss)
        print(f'Fine-tuning Epoch {epoch+1}, Average loss: {avg_loss:.4f}')
    
    fine_tuning_time = time.time() - start_time
    
    final_accuracy, final_conf_matrix = evaluate_model(model, test_loader, device)
    final_digit_accuracies = analyze_performance_by_digit(final_conf_matrix)
    
    with open('experiment_results.txt', 'w') as f:
        f.write("Experiment Results\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Initial training time: {initial_training_time:.2f} seconds\n")
        f.write(f"Fine-tuning time: {fine_tuning_time:.2f} seconds\n")
        f.write(f"Initial accuracy: {accuracy:.4f}\n")
        f.write(f"Final accuracy: {final_accuracy:.4f}\n")
        f.write(f"Worst digit initial accuracy: {digit_accuracies[worst_digit]:.4f}\n")
        f.write(f"Worst digit final accuracy: {final_digit_accuracies[worst_digit]:.4f}\n")
    
    print("\nFinal Results:")
    print(f"Initial training time: {initial_training_time:.2f} seconds")
    print(f"Fine-tuning time: {fine_tuning_time:.2f} seconds")
    print(f"Initial accuracy: {accuracy:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Worst digit initial accuracy: {digit_accuracies[worst_digit]:.4f}")
    print(f"Worst digit final accuracy: {final_digit_accuracies[worst_digit]:.4f}")
    
    plot_training_curves(initial_loss, fine_tuning_loss)

if __name__ == "__main__":
    main()