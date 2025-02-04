import os
import ray
import pandas as pd
import json
import torch
from torch_geometric.data import HeteroData
from model import train_gnn_parallel
from parameters import Parameters

def load_data():
    data_dir = 'data'
    with open(f"{data_dir}/lastfm_asia_features.json", "r") as f:
        features = json.load(f)
    target_df = pd.read_csv(f"{data_dir}/lastfm_asia_target.csv")
    edges_df = pd.read_csv(f"{data_dir}/lastfm_asia_edges.csv")
    return features, target_df, edges_df

def main():
    ray.init(address="auto")

    features, target_df, edges_df = load_data()

    # Extract unique user IDs
    user_ids = set(target_df['id'])

    # Create edges
    directed_edges = []  # LIKED edges (user -> user)
    for user, liked_artists in features.items():
        for artist in liked_artists:
            if int(artist) in user_ids:  # Ensure artist exists as a user
                directed_edges.append((int(user), int(artist)))

    # Convert edges to tensors
    connected_edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)  # Undirected
    liked_edge_index = torch.tensor(list(zip(*directed_edges)), dtype=torch.long)  # Directed

    graph_data = HeteroData()
    graph_data["user"].y = torch.tensor(target_df['target'].values, dtype=torch.long)
    graph_data["user", "CONNECTED", "user"].edge_index = connected_edge_index
    graph_data["user", "LIKED", "user"].edge_index = liked_edge_index

    # Initialize user embeddings dynamically
    num_users = len(user_ids)
    graph_data["user"].x = torch.nn.Embedding(num_users, Parameters.embedding_dim).weight
    
    # Run training with Ray
    future_model = train_gnn_parallel.remote(graph_data, embedding_dim=Parameters.embedding_dim, epochs=Parameters.epochs, lr=Parameters.lr)
    trained_model = ray.get(future_model)

    # Save model
    torch.save(trained_model.state_dict(), 'model/hetero_gnn_model.pth')

if __name__ == "__main__":
    main()