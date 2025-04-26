import pandas as pd
import numpy as np
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import os
import datetime

# Define the API sequence that your system uses
API_SEQUENCE = ["login", "get_user", "update_profile", "get_data", "logout"]

# Step 1: Upload and load data
def upload_and_load_data():
    print("Please provide the path to your logs JSON file:")
    filename = input()  # User provides the file path directly
    
    if not os.path.exists(filename):
        print("File not found.")
        return None

    try:
        # Read the file as JSON
        logs_df = pd.read_json(filename, lines=True)  # Assuming it's in JSON Lines format
        print(f"Successfully loaded {len(logs_df)} log entries from {filename} as JSONL")
    except:
        try:
            # If the above fails, try reading it as a regular JSON array
            logs_df = pd.read_json(filename)
            print(f"Successfully loaded {len(logs_df)} log entries from {filename} as JSON")
        except:
            print("Error loading JSON. Make sure your file contains valid JSON data.")
            return None

    # If you have existing anomaly data, upload it now or create synthetic labels
    print("\nDo you have an anomaly results CSV file? (y/n)")
    has_anomaly_file = input().lower().strip() == 'y'

    if has_anomaly_file:
        print("Please provide the path to your anomaly results CSV file:")
        anomaly_filename = input()
        if os.path.exists(anomaly_filename):
            anomaly_results_df = pd.read_csv(anomaly_filename)
            print(f"Successfully loaded {len(anomaly_results_df)} anomaly results from {anomaly_filename}")
        else:
            print("File not found.")
            anomaly_results_df = create_synthetic_labels(logs_df)
    else:
        print("Creating synthetic anomaly labels from your log data...")
        # Create synthetic labels based on basic rules
        anomaly_results_df = create_synthetic_labels(logs_df)

    return logs_df, anomaly_results_df

# Create synthetic labels for all log entries
def create_synthetic_labels(logs_df):
    results = []

    for _, log in logs_df.iterrows():
        # Simple rule-based anomaly detection
        is_error = log.get('response', {}).get('status_code', 200) >= 400
        is_slow = log.get('performance', {}).get('total_time_ms', 0) > 500
        high_cpu = log.get('resource_metrics', {}).get('cpu_utilization', 0) > 80

        # Determine if this is an anomaly
        is_anomaly = is_error or is_slow or high_cpu

        # Create a result entry for every log, not just anomalies
        result = {
            "timestamp": log.get('timestamp', pd.Timestamp.now().isoformat()),
            "user_id": log.get("operation", {}).get("user_id", "unknown"),
            "api": log.get("operation", {}).get("type", "unknown"),
            "environment": log.get("meta", {}).get("environment", "production"),
            "is_anomaly": is_anomaly,
            "api_impacts": {},
            "server_impacts": {},
            "user_blocked": is_error and is_slow,
            "next_user_impact": "🔴 HIGH" if (is_error and high_cpu) else "🟠 MED" if is_error else "🟡 LOW" if is_slow or high_cpu else "⚪ NONE"
        }

        # Add impacts for each API
        for api in API_SEQUENCE:
            if log.get("operation", {}).get("type") == api and is_error:
                result["api_impacts"][api] = "🔴 HIGH"
            elif log.get("operation", {}).get("type") == api and is_slow:
                result["api_impacts"][api] = "🟠 MED"
            else:
                result["api_impacts"][api] = "⚪ NONE"

        # Add server impacts
        result["server_impacts"] = {
            "server_1": "🔴 HIGH" if high_cpu else "⚪ NONE",
            "server_2": "🟠 MED" if is_slow else "⚪ NONE"
        }

        results.append(result)

    # Convert to DataFrame
    anomaly_df = pd.DataFrame(results)

    # Log counts of normal vs anomaly cases
    anomaly_count = anomaly_df['is_anomaly'].sum()
    print(f"Created {len(anomaly_df)} synthetic labels ({anomaly_count} anomalies, {len(anomaly_df) - anomaly_count} normal logs)")

    return anomaly_df

# Step 2: Data preparation function
def prepare_data(logs_df):
    # Extract features from logs
    features = []
    for _, log in logs_df.iterrows():
        # Extract features
        try:
            feature_vector = [
                log.get('response', {}).get('time_ms', 0) / 1000,
                (log.get('response', {}).get('status_code', 200) - 200) / 300,
                int(log.get('security', {}).get('mfa_used', True)),
                {'trusted': 2, 'normal': 1, 'suspicious': 0}.get(log.get('security', {}).get('ip_reputation', 'normal'), 1),
                log.get('security', {}).get('rate_limit', {}).get('remaining', 100) / 100,
                log.get('performance', {}).get('total_time_ms', 0) / 1000,
                log.get('performance', {}).get('db_query_time_ms', 0) / 1000 if log.get('performance', {}).get('db_query_time_ms') else 0,
                log.get('performance', {}).get('third_party_time_ms', 0) / 1000 if log.get('performance', {}).get('third_party_time_ms') else 0,
                log.get('resource_metrics', {}).get('cpu_utilization', 50) / 100,
                log.get('resource_metrics', {}).get('memory_utilization', 50) / 100,
                min(1.0, log.get('resource_metrics', {}).get('queue_depth', 0) / 20),
                min(1.0, log.get('request', {}).get('size_bytes', 0) / 2000),
                min(1.0, log.get('response', {}).get('size_bytes', 0) / 10000),
                1 if log.get('response', {}).get('status_code', 200) >= 400 else 0,
                1 if log.get('operation', {}).get('success', True) else 0,
                min(1.0, log.get('tracing', {}).get('session_failures', 0) / 5),
                min(1.0, log.get('tracing', {}).get('api_failures', 0) / 5),
                min(1.0, log.get('meta', {}).get('retry_count', 0) / 3)
            ]
            features.append(feature_vector)
        except Exception as e:
            print(f"Error extracting features from log: {e}")
            print(f"Log data: {log}")
            # Add a row of zeros as a placeholder
            features.append([0] * 18)

    return np.array(features)

# Step 3: Create training labels from existing system or synthetic data
def create_training_labels(logs_df, anomaly_results_df):
    # Map impact symbols to numerical values
    impact_map = {
        "🔴 HIGH": 0.9,
        "🟠 MED": 0.6,
        "🟡 LOW": 0.3,
        "⚪ NONE": 0.0
    }

    # Create comprehensive label vectors for GNN
    y_data = []

    # Extract labels from existing system's outputs or synthetic data
    for _, result in anomaly_results_df.iterrows():
        # Combine all impacts into a single vector for the GNN
        label_vector = []

        # API impacts - 1 value per API
        for api in API_SEQUENCE:
            if isinstance(result.get('api_impacts'), dict):
                impact_str = result.get('api_impacts', {}).get(api, "⚪ NONE")
            else:
                impact_str = "⚪ NONE"
            label_vector.append(impact_map.get(impact_str, 0.0))

        # Server impacts - 1 value per server
        for server in ["server_1", "server_2"]:
            if isinstance(result.get('server_impacts'), dict):
                impact_str = result.get('server_impacts', {}).get(server, "⚪ NONE")
            else:
                impact_str = "⚪ NONE"
            label_vector.append(impact_map.get(impact_str, 0.0))

        # User impacts
        label_vector.append(1.0 if result.get('user_blocked', False) else 0.0)

        # Next user impact
        next_impact_str = result.get('next_user_impact', "⚪ NONE")
        label_vector.append(impact_map.get(next_impact_str, 0.0))

        y_data.append(label_vector)

    return np.array(y_data)

# Define the GNN model for impact prediction
class ImpactGNN(nn.Module):
    def _init_(self, input_dim, hidden_dim, output_dim):
        super(ImpactGNN, self)._init_()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Function to create graph from features for GNN
def create_graph_data(features, labels=None):
    # Convert features to tensor
    x = torch.tensor(features, dtype=torch.float)

    # Create edges - we're using a simple fully connected graph here
    num_nodes = features.shape[0]
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create graph data
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
    else:
        data = Data(x=x, edge_index=edge_index)

    return data

# Step 4: Train models
def train_models(features, labels):
    # Ensure we have matching number of features and labels
    if len(features) != len(labels):
        print(f"Warning: Number of feature samples ({len(features)}) doesn't match label samples ({len(labels)})")
        min_samples = min(len(features), len(labels))
        features = features[:min_samples]
        labels = labels[:min_samples]
        print(f"Adjusted to use {min_samples} samples for training")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models dict
    models = {}

    # 1. Anomaly detection model with Isolation Forest
    print("Training anomaly detection model...")
    models['anomaly_detector'] = IsolationForest(contamination=0.2, random_state=42)
    models['anomaly_detector'].fit(X_train_scaled)

    # 2. Create graph data for GNN
    print("Creating graph data for GNN...")
    train_data = create_graph_data(X_train_scaled, y_train)
    test_data = create_graph_data(X_test_scaled, y_test)

    # 3. Train GNN for impact prediction
    print("Training GNN for impact prediction...")
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 64
    output_dim = y_train.shape[1]  # Number of outputs (API impacts + server impacts + user impacts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models['impact_gnn'] = ImpactGNN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(models['impact_gnn'].parameters(), lr=0.01)

    # Train the GNN
    models['impact_gnn'].train()
    train_data = train_data.to(device)

    print("Starting GNN training...")
    for epoch in range(100):
        optimizer.zero_grad()
        out = models['impact_gnn'](train_data)
        loss = F.mse_loss(out, train_data.y)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()

    print("GNN training complete")

    # Evaluate models
    models['impact_gnn'].eval()
    test_data = test_data.to(device)
    with torch.no_grad():
        pred = models['impact_gnn'](test_data)
        test_loss = F.mse_loss(pred, test_data.y)
        print(f"Test Loss: {test_loss.item():.4f}")

    return models, scaler, X_test_scaled, y_test

# Main execution flow
def main():
    print("==== Anomaly Detection with Isolation Forest + GNN ====")

    # 1. Upload and load data
    logs_df, anomaly_results_df = upload_and_load_data()
    if logs_df is None:
        return

    # 2. Prepare features and labels
    print("\nExtracting features from logs...")
    features = prepare_data(logs_df)
    print(f"Extracted {features.shape[1]} features from {features.shape[0]} log entries")

    print("\nCreating training labels...")
    labels = create_training_labels(logs_df, anomaly_results_df)

    # 3. Train models
    print("\nTraining ML models...")
    models, scaler, X_test, y_test = train_models(features, labels)

    # 4. Save models
    save_models(models, scaler)

    print("\nDone! The trained models have been saved and can be used for future log analysis.")

if __name__ == "__main__":
    main()