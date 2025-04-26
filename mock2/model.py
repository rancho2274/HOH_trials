import random
import uuid
import json
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch_geometric.nn as tg_nn
import torch.optim as optim
import networkx as nx

# Function to load and preprocess the logs
# Function to load and preprocess the logs
def load_and_preprocess_logs(file_path):
    logs = []
    with open(file_path, 'r') as f:
        for line in f:
            logs.append(json.loads(line.strip()))

    data = []
    for log in logs:
        # Use .get() to handle missing keys and provide default values
        operation_type = log['operation'].get('operation_type', 'unknown')  # Default to 'unknown' if missing
        data.append({
            'timestamp': log['timestamp'],
            'service_type': log['service']['type'],  # Updated to 'service' instead of 'api_service'
            'status_code': log['response']['status_code'],
            'response_time_ms': log['performance']['total_time_ms'],  # Updated to 'total_time_ms'
            'db_query_time_ms': log['performance']['db_query_time_ms'],
            'cpu_utilization': log['resource_metrics']['cpu_utilization'],
            'memory_utilization': log['resource_metrics']['memory_utilization'],
            'queue_depth': log['resource_metrics']['queue_depth'],
            'is_anomalous': 1 if log['response']['status_code'] in [504, 503, 500, 429] else 0,
            'request_id': log['request']['id'],
            'correlation_id': log['tracing']['correlation_id'],
            'previous_api': log['tracing']['previous_api'],
            'operation_type': operation_type,  # Safely using the 'get' method to avoid KeyError
            'user_id': log['operation']['user_id'],
            'tenant_id': log['operation']['tenant_id'],
        })

    return logs, data


# Function to preprocess logs for GNN
def preprocess_for_gnn(logs):
    G = nx.DiGraph()  # Directed graph to represent dependencies between APIs

    # Add nodes and edges for each log entry
    for log in logs:
        service_type = log['service_type']
        if service_type not in G:
            G.add_node(service_type)
        previous_api = log['tracing']['previous_api']
        if previous_api != service_type and previous_api is not None:
            G.add_edge(previous_api, service_type)

    # Convert graph to a format suitable for GNN (edge list and node features)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.rand((len(G.nodes), 5))  # 5 random features for each service
    data = Data(x=x, edge_index=edge_index)
    return data

# Isolation Forest Model for Anomaly Detection
def isolation_forest(logs):
    features = preprocess_for_iso_forest(logs)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(features_scaled)

    # Predict anomalies
    predictions = iso_forest.predict(features_scaled)
    anomalies = [1 if p == -1 else 0 for p in predictions]
    return anomalies

# GNN Model Definition
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = tg_nn.GCNConv(in_channels, 64)
        self.conv2 = tg_nn.GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Function to train the GNN model
def train_gnn(data, model, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)  # Cross-entropy loss for classification
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss {loss.item()}')

# Main function to run the training pipeline
def main():
    # Load and preprocess logs
    logs, data = load_and_preprocess_logs('generated_logs.jsonl')

    # Train Isolation Forest and get anomalies
    anomalies = isolation_forest(logs)
    print(f"Anomalies detected: {sum(anomalies)}")

    # Create graph data for GNN
    gnn_data = preprocess_for_gnn(logs)

    # Initialize GNN model
    gnn_model = GNN(in_channels=5, out_channels=2)  # 5 features, 2 output classes (normal/anomalous)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)

    # Train the GNN model
    train_gnn(gnn_data, gnn_model, optimizer)

if __name__ == "__main__":
    main()
