import json
import random
import datetime
import uuid
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import all mock generators
try:
    # Original API logs generator
    from try_module import generate_api_logs, save_logs_to_file, analyze_logs, generate_log_entry
    
    # Service API generator
    from service_api_generator import generate_service_logs, save_service_logs, generate_service_log_entry
    
    # Database API generator
    from database_api_generator import generate_db_logs, save_db_logs, generate_db_log_entry
    
    # Gateway API generator
    from gateway_api_generator import generate_gateway_logs, save_gateway_logs, generate_interconnected_logs, generate_gateway_log_entry, generate_gateway_request_flow
    
    # CDN API generator
    from cdn_api_generator import generate_cdn_request_flow, save_cdn_logs, generate_interconnected_cdn_logs
    
    # Auth API generator
    from hackohire.HOH.mock2.auth_api_generator import generate_auth_logs, save_auth_logs, generate_interconnected_auth_logs, generate_auth_flow
    
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import all mock generators - {e}")
    IMPORT_SUCCESS = False

def generate_all_interconnected_logs(num_flows=100, output_dir="interconnected_logs", anomaly_percentage=20):
    """Generate logs from all sources that are interconnected via correlation IDs"""
    
    if not IMPORT_SUCCESS:
        print("Error: Missing required modules. Please ensure all mock generators are available.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_flows} interconnected request flows across all services...")
    
    # Generate correlation IDs for all flows
    correlation_ids = [str(uuid.uuid4()) for _ in range(num_flows)]
    
    # Determine which flows will be anomalous
    num_anomalous_flows = int(num_flows * (anomaly_percentage / 100))
    anomalous_correlations = set(random.sample(correlation_ids, num_anomalous_flows))
    
    # Initialize collections for each log type
    api_logs = []
    gateway_logs = []
    service_logs = []
    db_logs = []
    cdn_logs = []
    auth_logs = []
    
    # Generate logs for each flow
    for i, correlation_id in enumerate(correlation_ids):
        print(f"Generating flow {i+1}/{num_flows} - Correlation ID: {correlation_id}")
        
        is_anomalous = correlation_id in anomalous_correlations
        anomaly_str = "ANOMALOUS" if is_anomalous else "normal"
        print(f"  Flow type: {anomaly_str}")
        
        # Generate a base timestamp for this flow
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
        
        # First, generate auth logs (usually the first step in a flow)
        flow_auth_logs = generate_auth_flow(
            base_time=base_time,
            correlation_id=correlation_id,
            is_anomalous=is_anomalous and random.random() < 0.3
        )
        auth_logs.extend(flow_auth_logs)
        
        # Add a small delay after auth
        current_time = base_time + datetime.timedelta(milliseconds=random.randint(50, 200))
        
        # Next, generate gateway logs (entry point to APIs)
        flow_gateway_logs = generate_gateway_request_flow(
            base_time=current_time,
            correlation_id=correlation_id,
            is_anomalous=is_anomalous and random.random() < 0.3
        )
        gateway_logs.extend(flow_gateway_logs)
        
        # Original API logs 
        num_api_logs = random.randint(2, 5)
        for j in range(num_api_logs):
            # Add a small delay
            api_time = current_time + datetime.timedelta(milliseconds=random.randint(10, 100) * j)
            
            api_log = generate_log_entry(
                timestamp=api_time,
                correlation_id=correlation_id,
                is_anomalous=is_anomalous and random.random() < 0.3
            )
            api_logs.append(api_log)
        
        # Generate service logs based on gateway calls
        for gateway_log in flow_gateway_logs:
            # Service logs usually happen after gateway processing
            service_time = datetime.datetime.fromisoformat(gateway_log["timestamp"]) + datetime.timedelta(milliseconds=random.randint(10, 100))
            
            # Each gateway call might trigger 1-3 service calls
            num_service_calls = random.randint(1, 3)
            for j in range(num_service_calls):
                # Small incremental delay between service calls
                current_service_time = service_time + datetime.timedelta(milliseconds=j * random.randint(5, 30))
                
                # Create service log connected to this gateway request
                service_log = generate_service_log_entry(
                    timestamp=current_service_time,
                    correlation_id=correlation_id,
                    is_anomalous=is_anomalous and random.random() < 0.3
                )
                service_logs.append(service_log)
                
                # Each service call might trigger 0-2 database operations
                num_db_ops = random.randint(0, 2)
                for k in range(num_db_ops):
                    db_time = current_service_time + datetime.timedelta(milliseconds=random.randint(5, 50))
                    
                    db_log = generate_db_log_entry(
                        timestamp=db_time,
                        correlation_id=correlation_id,
                        is_anomalous=is_anomalous and random.random() < 0.3
                    )
                    db_logs.append(db_log)
        
        # Generate CDN logs that might be triggered during the flow
        # CDN resources are usually loaded alongside or after API calls
        cdn_time = current_time + datetime.timedelta(milliseconds=random.randint(0, 100))
        
        # If this is a web flow, add page load resources
        if random.random() < 0.7:  # 70% of flows include CDN resources
            page_type = random.choice(["product", "category", "homepage", "checkout"])
            cdn_page_logs = generate_cdn_request_flow(
                base_time=cdn_time,
                correlation_id=correlation_id,
                page_type=page_type,
                is_anomalous=is_anomalous and random.random() < 0.3
            )
            cdn_logs.extend(cdn_page_logs)
    
    # Save all logs
    print("\nSaving all interconnected logs...")
    
    # Save original API logs
    api_logs_path = os.path.join(output_dir, "api_logs.json")
    with open(api_logs_path, 'w') as f:
        json.dump(api_logs, f, indent=2)
    print(f"Saved {len(api_logs)} original API logs to {api_logs_path}")
    
    # Save gateway logs
    gateway_logs_path = os.path.join(output_dir, "gateway_logs.json")
    with open(gateway_logs_path, 'w') as f:
        json.dump(gateway_logs, f, indent=2)
    print(f"Saved {len(gateway_logs)} gateway logs to {gateway_logs_path}")
    
    # Save service logs
    service_logs_path = os.path.join(output_dir, "service_logs.json")
    with open(service_logs_path, 'w') as f:
        json.dump(service_logs, f, indent=2)
    print(f"Saved {len(service_logs)} service logs to {service_logs_path}")
    
    # Save database logs
    db_logs_path = os.path.join(output_dir, "db_logs.json")
    with open(db_logs_path, 'w') as f:
        json.dump(db_logs, f, indent=2)
    print(f"Saved {len(db_logs)} database logs to {db_logs_path}")
    
    # Save CDN logs
    cdn_logs_path = os.path.join(output_dir, "cdn_logs.json")
    with open(cdn_logs_path, 'w') as f:
        json.dump(cdn_logs, f, indent=2)
    print(f"Saved {len(cdn_logs)} CDN logs to {cdn_logs_path}")
    
    # Save auth logs
    auth_logs_path = os.path.join(output_dir, "auth_logs.json")
    with open(auth_logs_path, 'w') as f:
        json.dump(auth_logs, f, indent=2)
    print(f"Saved {len(auth_logs)} auth logs to {auth_logs_path}")
    
    # Create summary information
    summary = {
        "total_flows": num_flows,
        "anomalous_flows": num_anomalous_flows,
        "normal_flows": num_flows - num_anomalous_flows,
        "anomaly_percentage": anomaly_percentage,
        "log_counts": {
            "api_logs": len(api_logs),
            "gateway_logs": len(gateway_logs),
            "service_logs": len(service_logs),
            "db_logs": len(db_logs),
            "cdn_logs": len(cdn_logs),
            "auth_logs": len(auth_logs),
            "total_logs": len(api_logs) + len(gateway_logs) + len(service_logs) + 
                           len(db_logs) + len(cdn_logs) + len(auth_logs)
        },
        "correlation_ids": correlation_ids,
        "anomalous_correlation_ids": list(anomalous_correlations)
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGenerated a total of {summary['log_counts']['total_logs']} logs across all services")
    print(f"Summary information saved to {summary_path}")
    
    return summary

def verify_correlation_coverage():
    """Check if correlation IDs link all the generated logs properly"""
    # Load logs from all sources
    try:
        with open("interconnected_logs/api_logs.json", 'r') as f:
            api_logs = json.load(f)
        
        with open("interconnected_logs/gateway_logs.json", 'r') as f:
            gateway_logs = json.load(f)
        
        with open("interconnected_logs/service_logs.json", 'r') as f:
            service_logs = json.load(f)
        
        with open("interconnected_logs/db_logs.json", 'r') as f:
            db_logs = json.load(f)
        
        with open("interconnected_logs/cdn_logs.json", 'r') as f:
            cdn_logs = json.load(f)
        
        with open("interconnected_logs/auth_logs.json", 'r') as f:
            auth_logs = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not load all log files - {e}")
        return
    
    # Extract correlation IDs from each source
    api_correlations = set()
    for log in api_logs:
        if "correlation_id" in log:
            api_correlations.add(log["correlation_id"])
    
    gateway_correlations = set()
    for log in gateway_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            gateway_correlations.add(log["tracing"]["correlation_id"])
    
    service_correlations = set()
    for log in service_logs:
        if "correlation" in log and "id" in log["correlation"]:
            service_correlations.add(log["correlation"]["id"])
    
    db_correlations = set()
    for log in db_logs:
        if "correlation_id" in log:
            db_correlations.add(log["correlation_id"])
    
    cdn_correlations = set()
    for log in cdn_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            cdn_correlations.add(log["tracing"]["correlation_id"])
    
    auth_correlations = set()
    for log in auth_logs:
        if "tracing" in log and "correlation_id" in log["tracing"]:
            auth_correlations.add(log["tracing"]["correlation_id"])
    
    # Find correlation IDs that appear in all systems
    all_correlations = (api_correlations & gateway_correlations & 
                        service_correlations & db_correlations & 
                        cdn_correlations & auth_correlations)
    
    # Print results
    print("\n=== Correlation ID Coverage Analysis ===")
    print(f"API logs correlation IDs: {len(api_correlations)}")
    print(f"Gateway logs correlation IDs: {len(gateway_correlations)}")
    print(f"Service logs correlation IDs: {len(service_correlations)}")
    print(f"Database logs correlation IDs: {len(db_correlations)}")
    print(f"CDN logs correlation IDs: {len(cdn_correlations)}")
    print(f"Auth logs correlation IDs: {len(auth_correlations)}")
    print(f"\nCorrelation IDs present in all systems: {len(all_correlations)}")
    
    # Check overlap between pairs of systems
    print("\n=== Pairwise Correlation Coverage ===")
    pairs = [
        ("API-Gateway", api_correlations & gateway_correlations),
        ("API-Service", api_correlations & service_correlations),
        ("API-Database", api_correlations & db_correlations),
        ("API-CDN", api_correlations & cdn_correlations),
        ("API-Auth", api_correlations & auth_correlations),
        ("Gateway-Service", gateway_correlations & service_correlations),
        ("Gateway-Database", gateway_correlations & db_correlations),
        ("Gateway-CDN", gateway_correlations & cdn_correlations),
        ("Gateway-Auth", gateway_correlations & auth_correlations),
        ("Service-Database", service_correlations & db_correlations),
        ("Service-CDN", service_correlations & cdn_correlations),
        ("Service-Auth", service_correlations & auth_correlations),
        ("Database-CDN", db_correlations & cdn_correlations),
        ("Database-Auth", db_correlations & auth_correlations),
        ("CDN-Auth", cdn_correlations & auth_correlations)
    ]
    
    for pair_name, common_ids in pairs:
        print(f"{pair_name}: {len(common_ids)} shared correlation IDs")
    
    return all_correlations

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate interconnected logs
    print("=== Generating Interconnected Logs Across All API Systems ===\n")
    summary = generate_all_interconnected_logs(num_flows=100, anomaly_percentage=20)
    
    # Verify correlation coverage
    print("\n=== Verifying Correlation ID Coverage ===")
    all_correlations = verify_correlation_coverage()
    
    print("\nLog generation complete!")
    print("You can now use the integrated_logs_analyzer.py script to analyze these interconnected logs:")
    print("python integrated_logs_analyzer.py --gateway-logs interconnected_logs/gateway_logs.json " +
          "--service-logs interconnected_logs/service_logs.json --db-logs interconnected_logs/db_logs.json " +
          "--output-dir analysis_results")