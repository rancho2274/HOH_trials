import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import argparse
import os

def load_logs(file_path):
    """Load API logs from a JSON or CSV file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Parse complex JSON columns
        for col in ['request_headers', 'request_body']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_logs(logs):
    """Convert logs to a pandas DataFrame and preprocess for model input"""
    # Convert to DataFrame
    df = pd.DataFrame(logs)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Calculate request body size
    df['request_body_size'] = df['request_body'].apply(
        lambda x: len(json.dumps(x)) if isinstance(x, (dict, list)) else 
                 (len(str(x)) if x is not None else 0)
    )
    
    # Count request headers
    df['num_headers'] = df['request_headers'].apply(
        lambda x: len(x) if isinstance(x, dict) else 0
    )
    
    # Extract error information
    df['has_error'] = df['error_message'].apply(lambda x: 1 if x is not None else 0)
    
    # Map status code to categories
    df['status_category'] = df['status_code'].apply(
        lambda x: 'success' if 200 <= x < 300 else
                  'client_error' if 400 <= x < 500 else
                  'server_error' if 500 <= x < 600 else 'other'
    )
    
    # Create numerical and categorical feature lists
    numerical_features = [
        'response_time', 
        'response_size', 
        'request_body_size',
        'num_headers',
        'hour',
        'day_of_week'
    ]
    
    categorical_features = [
        'http_method',
        'endpoint_path',
        'status_category',
        'has_error'
    ]
    
    # Create feature dataframe
    features_df = df[numerical_features + categorical_features].copy()
    
    # Drop rows with NaN values or replace with appropriate defaults
    features_df = features_df.fillna({
        'response_time': df['response_time'].median(),
        'response_size': 0,
        'request_body_size': 0,
        'num_headers': 0,
        'has_error': 0
    })
    
    # Return the original dataframe and the features dataframe
    return df, features_df, numerical_features, categorical_features

def build_preprocessing_pipeline(numerical_features, categorical_features):
    """Build a scikit-learn preprocessing pipeline for the features"""
    
    # Define preprocessing for numerical features
    numerical_transformer = StandardScaler()
    
    # Define preprocessing for categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0  # Return dense array for better compatibility with Isolation Forest
    )
    
    return preprocessor

def train_isolation_forest(X, contamination=0.1, random_state=42):
    """Train an Isolation Forest model"""
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        max_features=1.0
    )
    model.fit(X)
    return model

def evaluate_model(df, predictions, actual_label_column=None):
    """Evaluate the model's performance"""
    # Add predictions to the dataframe
    df['anomaly_score'] = predictions
    df['predicted_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    
    # If we have actual labels, compute metrics
    if actual_label_column and actual_label_column in df.columns:
        print("\n=== Model Evaluation ===")
        true_labels = df[actual_label_column].astype(int)
        pred_labels = df['predicted_anomaly']
        
        # Print confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print("Confusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))
        
        # Return predictions and actual labels for further analysis
        return df[['anomaly_score', 'predicted_anomaly', actual_label_column]]
    
    return df[['anomaly_score', 'predicted_anomaly']]

def visualize_anomalies(df, title='API Log Anomalies', output_dir='.'):
    """Visualize the anomalies in the dataset"""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Plot response time vs response size with anomalies highlighted
    ax = axes[0]
    scatter = ax.scatter(
        df['response_time'], 
        df['response_size'],
        c=df['predicted_anomaly'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='w',
        s=50
    )
    ax.set_title('Response Time vs Response Size')
    ax.set_xlabel('Response Time (seconds)')
    ax.set_ylabel('Response Size (bytes)')
    ax.grid(True, linestyle='--', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Anomaly")
    ax.add_artist(legend1)
    
    # 2. Plot status code distribution with anomalies
    ax = axes[1]
    status_anomaly = pd.crosstab(df['status_code'], df['predicted_anomaly'])
    status_anomaly.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Status Code Distribution')
    ax.set_xlabel('Status Code')
    ax.set_ylabel('Count')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 3. Plot endpoint distribution with anomalies
    ax = axes[2]
    endpoint_anomaly = pd.crosstab(df['endpoint_path'], df['predicted_anomaly'])
    endpoint_anomaly.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Endpoint Distribution')
    ax.set_xlabel('Count')
    ax.set_ylabel('Endpoint')
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # 4. Plot time of day distribution with anomalies
    ax = axes[3]
    hour_anomaly = pd.crosstab(df['hour'], df['predicted_anomaly'])
    hour_anomaly.plot(kind='line', ax=ax, marker='o', colormap='viridis')
    ax.set_title('Time of Day Distribution')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Count')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and add main title
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'api_log_anomalies.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

def analyze_anomalies(df):
    """Analyze the detected anomalies to gain insights"""
    anomalies = df[df['predicted_anomaly'] == 1].copy()
    normal = df[df['predicted_anomaly'] == 0].copy()
    
    print(f"\n=== Anomaly Analysis ===")
    print(f"Total logs: {len(df)}")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
    
    # Analyze status codes in anomalies
    print("\nStatus Code Distribution in Anomalies:")
    status_counts = anomalies['status_code'].value_counts().sort_index()
    for status, count in status_counts.items():
        print(f"  HTTP {status}: {count} ({count/len(anomalies)*100:.2f}%)")
    
    # Analyze response times
    print("\nResponse Time Statistics:")
    print(f"  Normal logs avg: {normal['response_time'].mean():.4f}s, median: {normal['response_time'].median():.4f}s")
    print(f"  Anomalous logs avg: {anomalies['response_time'].mean():.4f}s, median: {anomalies['response_time'].median():.4f}s")
    
    # Analyze endpoints with most anomalies
    print("\nTop Endpoints with Anomalies:")
    endpoint_counts = anomalies['endpoint_path'].value_counts().head(5)
    for endpoint, count in endpoint_counts.items():
        print(f"  {endpoint}: {count} ({count/len(anomalies)*100:.2f}%)")
    
    # Analyze error messages
    if 'error_message' in anomalies.columns:
        error_counts = anomalies['error_message'].dropna().value_counts().head(5)
        if not error_counts.empty:
            print("\nTop Error Messages in Anomalies:")
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
    
    # Analyze correlation IDs with multiple anomalies
    if 'correlation_id' in anomalies.columns:
        corr_counts = anomalies['correlation_id'].value_counts()
        multi_anomaly_flows = corr_counts[corr_counts > 1]
        if not multi_anomaly_flows.empty:
            print(f"\nRequest Flows with Multiple Anomalies: {len(multi_anomaly_flows)}")
            print(f"Top correlation IDs with multiple anomalies:")
            for corr_id, count in multi_anomaly_flows.head(5).items():
                print(f"  {corr_id}: {count} anomalies")
    
    return anomalies

def save_anomalies(anomalies, output_path='anomalies.json'):
    """Save detected anomalies to a file"""
    # Convert timestamp to string for JSON serialization
    if 'timestamp' in anomalies.columns:
        anomalies['timestamp'] = anomalies['timestamp'].astype(str)
    
    # Save to file
    anomalies.to_json(output_path, orient='records', indent=2)
    print(f"Anomalies saved to {output_path}")

def extract_connected_apis(df):
    """Extract information about connected APIs based on correlation IDs"""
    if 'correlation_id' not in df.columns:
        print("Correlation ID not found in logs. Cannot extract API connections.")
        return None
    
    # Group by correlation ID
    connected_requests = df.groupby('correlation_id').filter(lambda x: len(x) > 1)
    
    # Count unique correlation IDs
    flow_count = connected_requests['correlation_id'].nunique()
    print(f"\n=== API Connection Analysis ===")
    print(f"Found {flow_count} API request flows (multiple requests with same correlation ID)")
    
    # Extract common sequences
    flows = []
    for corr_id, group in connected_requests.groupby('correlation_id'):
        # Sort by timestamp to get the sequence
        sorted_group = group.sort_values('timestamp')
        
        # Extract the sequence of endpoints
        sequence = sorted_group['endpoint_path'].tolist()
        
        # Record the flow
        flows.append({
            'correlation_id': corr_id,
            'sequence': sequence,
            'contains_anomaly': sorted_group['predicted_anomaly'].sum() > 0,
            'request_count': len(sequence)
        })
    
    # Convert to DataFrame for analysis
    flows_df = pd.DataFrame(flows)
    
    # Print some statistics
    if not flows_df.empty:
        print(f"Average requests per flow: {flows_df['request_count'].mean():.2f}")
        print(f"Flows containing anomalies: {flows_df['contains_anomaly'].sum()} ({flows_df['contains_anomaly'].mean()*100:.2f}%)")
        
        # Find common sequences
        sequence_str = flows_df['sequence'].apply(lambda x: ' -> '.join(x))
        common_sequences = sequence_str.value_counts().head(5)
        
        print("\nMost common API call sequences:")
        for seq, count in common_sequences.items():
            print(f"  {seq}: {count} occurrences")
    
    return flows_df

def update_model_with_feedback(model, X, anomalies_df, feedback_column='confirmed_anomaly'):
    """Update the model based on user feedback about anomalies"""
    if feedback_column not in anomalies_df.columns:
        print("No feedback column found. Cannot update model.")
        return model
    
    # Get indices of confirmed anomalies
    confirmed_indices = anomalies_df[anomalies_df[feedback_column] == True].index
    
    if len(confirmed_indices) == 0:
        print("No confirmed anomalies found. Model remains unchanged.")
        return model
    
    print(f"Updating model with {len(confirmed_indices)} confirmed anomalies...")
    
    # Extract the feature vectors for confirmed anomalies
    X_confirmed = X[confirmed_indices]
    
    # Create a new model with a higher weight for confirmed anomalies
    # (This is a simplified approach; in a real system, you might use more sophisticated methods)
    new_model = IsolationForest(
        contamination=model.contamination,
        random_state=model.random_state,
        n_estimators=model.n_estimators,
        max_samples=model.max_samples
    )
    
    # Combine original training data with confirmed anomalies (weighted)
    X_augmented = np.vstack([X, np.repeat(X_confirmed, 3, axis=0)])
    
    # Train the new model
    new_model.fit(X_augmented)
    
    print("Model updated successfully.")
    return new_model

def main():
    parser = argparse.ArgumentParser(description='API Log Anomaly Detection')
    parser.add_argument('--input', '-i', required=True, help='Path to the input log file (JSON or CSV)')
    parser.add_argument('--output', '-o', default='anomalies.json', help='Path to save detected anomalies')
    parser.add_argument('--contamination', '-c', type=float, default=0.1, help='Expected proportion of anomalies (0.0-0.5)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations')
    parser.add_argument('--label-column', '-l', help='Column name for actual anomaly labels (if available)')
    parser.add_argument('--output-dir', '-d', default='.', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess logs
    print(f"Loading logs from {args.input}...")
    logs = load_logs(args.input)
    print(f"Loaded {len(logs)} log entries.")
    
    print("Preprocessing logs...")
    df, features_df, numerical_features, categorical_features = preprocess_logs(logs)
    
    # Build preprocessing pipeline
    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Transform features
    print("Transforming features...")
    X = preprocessor.fit_transform(features_df)
    
    # Train Isolation Forest model
    print(f"Training Isolation Forest model (contamination={args.contamination})...")
    model = train_isolation_forest(X, contamination=args.contamination)
    
    # Make predictions
    print("Detecting anomalies...")
    predictions = model.predict(X)
    
    # Evaluate model if labels are available
    if args.label_column and args.label_column in df.columns:
        evaluate_model(df, predictions, args.label_column)
    else:
        # Just add predictions to the dataframe
        df['anomaly_score'] = predictions
        df['predicted_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    
    # Analyze anomalies
    anomalies = analyze_anomalies(df)
    
    # Extract connected APIs information
    api_flows = extract_connected_apis(df)
    
    # Save anomalies
    output_path = os.path.join(args.output_dir, args.output)
    save_anomalies(anomalies, output_path)
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_anomalies(df, output_dir=args.output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()