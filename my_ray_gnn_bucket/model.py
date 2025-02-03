import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ParallelHeteroGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParallelHeteroGNN, self).__init__()
        self.conv_connected = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.conv_liked = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)  # Combining both embeddings

    def forward(self, x, edge_index_connected, edge_index_liked):
        x_connected = self.conv_connected(x, edge_index_connected).relu()
        x_liked = self.conv_liked(x, edge_index_liked).relu()
        
        # Combine embeddings from both edge types
        x_combined = torch.cat([x_connected, x_liked], dim=1)
        return self.fc(x_combined)
    
@ray.remote
class GNNTrainer:
    def __init__(self, model, edge_type):
        self.model = model
        self.edge_type = edge_type

    def compute_forward(self, x, edge_index):
        return self.model(x, edge_index).detach()
    

@ray.remote
def train_gnn_parallel(graph_data, embedding_dim, epochs=100, lr=0.01):
    model = ParallelHeteroGNN(input_dim=embedding_dim, hidden_dim=64, output_dim=18)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # use weighted loss to minimize the impact of imbalanced dataset
    
    criterion = nn.CrossEntropyLoss()

    train_idx, test_idx = train_test_split(range(len(graph_data["user"].y)), test_size=0.2, random_state=42)
    
    # Create Ray actors for both edge types
    connected_trainer = GNNTrainer.remote(model.conv_connected, "CONNECTED")
    liked_trainer = GNNTrainer.remote(model.conv_liked, "LIKED")

    train_losses, test_losses = [], []

    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Parallel execution of forward propagation
        future_connected = connected_trainer.compute_forward.remote(
            graph_data["user"].x, graph_data["user", "CONNECTED", "user"].edge_index
        )
        future_liked = liked_trainer.compute_forward.remote(
            graph_data["user"].x, graph_data["user", "LIKED", "user"].edge_index
        )

        x_connected, x_liked = ray.get([future_connected, future_liked])
        x_combined = torch.cat([x_connected, x_liked], dim=1)

        out = model.fc(x_combined)
        loss = criterion(out[train_idx], graph_data["user"].y[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_out = model.fc(torch.cat(ray.get([connected_trainer.compute_forward.remote(
                graph_data["user"].x, graph_data["user", "CONNECTED", "user"].edge_index
            ), liked_trainer.compute_forward.remote(
                graph_data["user"].x, graph_data["user", "LIKED", "user"].edge_index
            )]), dim=1))

            test_loss = criterion(test_out[test_idx], graph_data["user"].y[test_idx])
            test_losses.append(test_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")

    end = time.time()

    print(f'Time taken: {end - start} seconds')

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/hetero_gnn_model.pth")
    
    # Plot train and test loss
    os.makedirs("result", exist_ok=True)
    plt.figure()
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("result/train_test_loss.jpg")
    plt.close()
    
    # Confusion Matrix
    model.eval()
    predictions = model(
        graph_data["user"].x, 
        graph_data["user", "CONNECTED", "user"].edge_index,
        graph_data["user", "LIKED", "user"].edge_index
    ).argmax(dim=1)
    cm = confusion_matrix(graph_data["user"].y[test_idx].cpu().numpy(), predictions[test_idx].cpu().numpy())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("result/confusion_matrix.jpg")
    plt.close()
    
    return model