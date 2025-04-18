import json
import random
import datetime
import uuid
import time
from faker import Faker
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize Faker for generating realistic data
fake = Faker()

# Define constants specific to database API
DB_TYPES = ["mysql", "postgres", "mongodb", "cassandra", "redis"]
DB_OPERATIONS = ["query", "insert", "update", "delete", "aggregate", "index", "transaction"]
DB_COLLECTIONS = ["users", "products", "orders", "payments", "logs", "sessions", "metrics"]

# Database status codes
DB_STATUS_CODES = {
    0: "Success",
    1: "Warning",
    2: "Error",
    3: "Fatal Error",
    4: "Timeout",
    5: "Connection Error",
    6: "Authentication Error",
    7: "Permission Error",
    8: "Constraint Violation",
    9: "Syntax Error",
    10: "Data Integrity Error",
    11: "Lock Timeout",
    12: "Transaction Error",
    13: "Resource Exhausted",
    14: "Statement Limit Exceeded"
}

# Database error messages
DB_ERROR_MESSAGES = {
    1: ["Slow query warning", "Index not optimized", "Partial result returned"],
    2: ["Query execution error", "Connection pool exhausted", "Lock contention detected"],
    3: ["Database connection lost", "Out of memory error", "Disk space full"],
    4: ["Query timeout after {0} ms", "Connection timeout", "Lock wait timeout"],
    5: ["Cannot connect to database server", "Database server unreachable", "Connection refused"],
    6: ["Invalid credentials", "Authentication failed", "Database access denied"],
    7: ["Insufficient privileges for operation", "User lacks permission", "Read-only access violation"],
    8: ["Unique constraint violation", "Foreign key constraint failed", "Check constraint violation"],
    9: ["SQL syntax error", "Invalid collection name", "Malformed query"],
    10: ["Data truncation error", "Invalid data format", "Type conversion failed"],
    11: ["Transaction deadlock detected", "Lock acquisition timeout", "Row lock contention"],
    12: ["Transaction rollback", "Transaction aborted", "Savepoint does not exist"],
    13: ["Connection limit reached", "Too many concurrent queries", "Worker thread pool exhausted"],
    14: ["Query complexity too high", "Too many joins in query", "Subquery limit exceeded"]
}

# SQL syntax keywords for realistic query generation
SQL_KEYWORDS = ["SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "INNER JOIN", "GROUP BY", 
                "ORDER BY", "HAVING", "LIMIT", "OFFSET", "INSERT INTO", "VALUES", 
                "UPDATE", "SET", "DELETE FROM", "CREATE", "ALTER", "DROP", "INDEX"]

# MongoDB operation keywords
MONGO_KEYWORDS = ["find", "findOne", "insertOne", "insertMany", "updateOne", "updateMany", 
                 "deleteOne", "deleteMany", "aggregate", "count", "distinct", "bulkWrite"]

# Database servers
DB_SERVERS = {
    "mysql": ["mysql-primary", "mysql-replica-1", "mysql-replica-2", "mysql-analytics"],
    "postgres": ["postgres-main", "postgres-reporting", "postgres-archive", "postgres-replica"],
    "mongodb": ["mongodb-shard-1", "mongodb-shard-2", "mongodb-shard-3", "mongodb-config"],
    "cassandra": ["cassandra-node-1", "cassandra-node-2", "cassandra-node-3", "cassandra-seed"],
    "redis": ["redis-cache", "redis-session", "redis-queue", "redis-pubsub"]
}

def generate_db_instance_id(db_type):
    """Generate realistic database instance ID"""
    server = random.choice(DB_SERVERS.get(db_type, [f"{db_type}-server"]))
    return f"{server}-{random.randint(1, 5)}"

def generate_db_query(db_type, operation, collection, is_anomalous=False):
    """Generate realistic database query strings"""
    query = ""
    
    if db_type in ["mysql", "postgres"]:
        # SQL-based query
        if operation == "query":
            fields = ", ".join(random.sample(["id", "name", "email", "status", "created_at", "updated_at", "type", "category"], 
                                          random.randint(1, 5)))
            
            query = f"SELECT {fields} FROM {collection}"
            
            # Add WHERE clause with some probability
            if random.random() < 0.7:
                where_field = random.choice(["id", "status", "type", "created_at"])
                if where_field == "id":
                    query += f" WHERE id = '{uuid.uuid4()}'"
                elif where_field == "created_at":
                    days = random.randint(1, 30)
                    query += f" WHERE created_at > NOW() - INTERVAL '{days} days'"
                else:
                    query += f" WHERE {where_field} = '{random.choice(['active', 'pending', 'completed', 'archived'])}'"
            
            # Add ORDER BY with some probability
            if random.random() < 0.5:
                order_field = random.choice(["created_at", "updated_at", "id"])
                order_dir = random.choice(["ASC", "DESC"])
                query += f" ORDER BY {order_field} {order_dir}"
            
            # Add LIMIT with some probability
            if random.random() < 0.6:
                limit = random.randint(10, 1000)
                query += f" LIMIT {limit}"
                
                # Sometimes add OFFSET
                if random.random() < 0.3:
                    offset = random.randint(0, 5000)
                    query += f" OFFSET {offset}"
            
            # Introduce anomalies if needed
            if is_anomalous:
                if random.random() < 0.3:
                    # Missing WHERE clause on large table - potential full table scan
                    query = f"SELECT * FROM {collection}"
                elif random.random() < 0.6:
                    # Bad JOIN condition
                    other_table = random.choice([t for t in DB_COLLECTIONS if t != collection])
                    query += f" LEFT JOIN {other_table} ON 1=1"  # Cartesian product
                else:
                    # Extremely large LIMIT
                    query = query.replace("LIMIT", "LIMIT 1000000")
        
        elif operation == "insert":
            columns = random.sample(["id", "name", "email", "status", "created_at", "type"], random.randint(3, 6))
            columns_str = ", ".join(columns)
            
            values = []
            for col in columns:
                if col == "id":
                    values.append(f"'{uuid.uuid4()}'")
                elif col == "name":
                    values.append(f"'{fake.name()}'")
                elif col == "email":
                    values.append(f"'{fake.email()}'")
                elif col == "status":
                    values.append(f"'{random.choice(['active', 'inactive', 'pending'])}'")
                elif col == "created_at":
                    values.append("NOW()")
                elif col == "type":
                    values.append(f"'{random.choice(['user', 'admin', 'guest'])}'")
            
            values_str = ", ".join(values)
            query = f"INSERT INTO {collection} ({columns_str}) VALUES ({values_str})"
            
            # Introduce anomalies if needed
            if is_anomalous and random.random() < 0.5:
                # Missing column in VALUES clause
                values_parts = values_str.split(", ")
                if len(values_parts) > 1:
                    values_str = ", ".join(values_parts[:-1])
                    query = f"INSERT INTO {collection} ({columns_str}) VALUES ({values_str})"
        
        elif operation == "update":
            update_cols = random.sample(["name", "email", "status", "updated_at", "type"], random.randint(1, 3))
            set_clauses = []
            
            for col in update_cols:
                if col == "name":
                    set_clauses.append(f"name = '{fake.name()}'")
                elif col == "email":
                    set_clauses.append(f"email = '{fake.email()}'")
                elif col == "status":
                    set_clauses.append(f"status = '{random.choice(['active', 'inactive', 'pending'])}'")
                elif col == "updated_at":
                    set_clauses.append("updated_at = NOW()")
                elif col == "type":
                    set_clauses.append(f"type = '{random.choice(['user', 'admin', 'guest'])}'")
            
            set_clause = ", ".join(set_clauses)
            query = f"UPDATE {collection} SET {set_clause} WHERE id = '{uuid.uuid4()}'"
            
            # Introduce anomalies if needed
            if is_anomalous and random.random() < 0.5:
                # Missing WHERE clause - updates all rows!
                query = f"UPDATE {collection} SET {set_clause}"
        
        elif operation == "delete":
            query = f"DELETE FROM {collection} WHERE id = '{uuid.uuid4()}'"
            
            # Introduce anomalies if needed
            if is_anomalous and random.random() < 0.6:
                # Missing WHERE clause - deletes all rows!
                query = f"DELETE FROM {collection}"
        
    elif db_type == "mongodb":
        # MongoDB-style query
        if operation in ["query", "find"]:
            # Simple find query
            if random.random() < 0.7:
                query = f"db.{collection}.find("
                if random.random() < 0.8:  # with filter
                    field = random.choice(["status", "type", "category"])
                    value = random.choice(["active", "inactive", "pending", "user", "admin", "product"])
                    query += f"{{{field}: '{value}'}}"
                query += ")"
                
                # Add projection with some probability
                if random.random() < 0.5:
                    fields = random.sample(["_id", "name", "email", "created_at"], random.randint(1, 3))
                    projection = "{" + ", ".join([f"'{field}': 1" for field in fields]) + "}"
                    query += f".project({projection})"
                
                # Add sort with some probability
                if random.random() < 0.4:
                    sort_field = random.choice(["created_at", "updated_at", "name"])
                    sort_dir = random.choice([1, -1])
                    query += f".sort({{'{sort_field}': {sort_dir}}})"
                
                # Add limit with some probability
                if random.random() < 0.6:
                    limit = random.randint(10, 100)
                    query += f".limit({limit})"
            else:
                # Aggregation pipeline
                query = f"db.{collection}.aggregate(["
                stages = []
                
                # Match stage
                if random.random() < 0.8:
                    field = random.choice(["status", "type", "category"])
                    value = random.choice(["active", "inactive", "pending", "user", "admin", "product"])
                    stages.append(f"{{$match: {{{field}: '{value}'}}}}")
                
                # Group stage
                if random.random() < 0.6:
                    group_field = random.choice(["status", "type", "category"])
                    stages.append(f"{{$group: {{_id: '${group_field}', count: {{$sum: 1}}}}}}")
                
                # Sort stage
                if random.random() < 0.5:
                    sort_field = "count" if "group" in str(stages) else random.choice(["created_at", "name"])
                    sort_dir = random.choice([1, -1])
                    stages.append(f"{{$sort: {{'{sort_field}': {sort_dir}}}}}")
                
                query += ", ".join(stages) + "])"
            
            # Introduce anomalies if needed
            if is_anomalous:
                if random.random() < 0.3:
                    # Missing index hint - potential full collection scan
                    query = f"db.{collection}.find({{}})"
                elif random.random() < 0.6:
                    # Complex query with no projection
                    query = f"db.{collection}.find({{$or: [{{status: 'active'}}, {{status: 'pending'}}]}}).sort({{created_at: -1}})"
                else:
                    # Too many stages in aggregation
                    query = f"db.{collection}.aggregate([{{$match: {{}}}}])"
                    for i in range(random.randint(5, 10)):
                        query = query[:-2] + f", {{$project: {{field{i}: 1}}}}" + "])"
        
        elif operation in ["insert", "insertOne"]:
            fields = {
                "_id": str(uuid.uuid4()),
                "name": fake.name(),
                "email": fake.email(),
                "status": random.choice(["active", "inactive", "pending"]),
                "created_at": "new Date()"
            }
            query = f"db.{collection}.insertOne({json.dumps(fields)})"
            
            # Introduce anomalies if needed
            if is_anomalous and random.random() < 0.5:
                # Missing required field or duplicate key
                if random.random() < 0.5:
                    # Duplicate key (simplified simulation)
                    query = f"db.{collection}.insertOne({{_id: 'duplicate_id', name: '{fake.name()}'}})"
                else:
                    # Missing required field (assuming 'name' is required)
                    del fields["name"]
                    query = f"db.{collection}.insertOne({json.dumps(fields)})"
        
        elif operation in ["update", "updateOne", "updateMany"]:
            field_to_update = random.choice(["status", "email", "updated_at", "type"])
            new_value = ("new Date()" if field_to_update == "updated_at" else 
                          f"'{fake.email()}'" if field_to_update == "email" else
                          f"'{random.choice(['active', 'inactive', 'pending'])}'")
            
            query = f"db.{collection}.updateOne({{_id: '{uuid.uuid4()}'}}, {{$set: {{{field_to_update}: {new_value}}}}})"
            
            # Introduce anomalies if needed
            if is_anomalous and random.random() < 0.5:
                # Update without criteria - updates many documents
                query = f"db.{collection}.updateMany({{}}, {{$set: {{{field_to_update}: {new_value}}}}})"
    elif db_type == "redis":
        # Redis commands
        key = f"{collection}:{uuid.uuid4()}"
        
        if operation == "query":
            query = f"GET {key}"
        elif operation == "insert":
            value = json.dumps({
                "id": str(uuid.uuid4()),
                "name": fake.name(),
                "timestamp": int(time.time())
            })
            query = f"SET {key} {value} EX 3600"  # 1 hour expiry
        elif operation == "update":
            field = random.choice(["name", "status", "score"])
            value = fake.name() if field == "name" else random.choice(["active", "inactive"]) if field == "status" else random.randint(1, 100)
            query = f"HSET {key} {field} {value}"
        elif operation == "delete":
            query = f"DEL {key}"
        
        # Introduce anomalies if needed
        if is_anomalous and random.random() < 0.5:
            if operation == "insert":
                # Missing expiry - potential memory leak
                query = f"SET {key} {value}"
            elif operation == "query":
                # Wrong data type operation
                query = f"HGETALL {key}"  # Trying to get hash fields from string key
    
    return query

def generate_db_log_entry(timestamp=None, db_type=None, operation=None, 
                       correlation_id=None, is_anomalous=False):
    """Generate a database API log entry"""
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
    
    # Select random database type if not provided
    if db_type is None:
        db_type = random.choice(DB_TYPES)
    
    # Select random operation if not provided
    if operation is None:
        operation = random.choice(DB_OPERATIONS)
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Select random collection/table
    collection = random.choice(DB_COLLECTIONS)
    
    # Generate database instance ID
    instance_id = generate_db_instance_id(db_type)
    
    # Generate query string
    query = generate_db_query(db_type, operation, collection, is_anomalous)
    
    # Calculate query execution time based on operation and anomaly
    if is_anomalous and random.random() < 0.7:
        # Anomalous queries tend to be much slower
        execution_time = random.uniform(1.0, 30.0)  # 1-30 seconds
    else:
        if operation in ["query", "aggregate"]:
            # Read operations have variable times
            execution_time = random.uniform(0.005, 1.0)  # 5ms-1s
        elif operation in ["insert", "update", "delete"]:
            # Write operations are usually faster for single records
            execution_time = random.uniform(0.002, 0.5)  # 2ms-500ms
        elif operation == "transaction":
            # Transactions take longer
            execution_time = random.uniform(0.01, 2.0)  # 10ms-2s
        elif operation == "index":
            # Indexing is slow
            execution_time = random.uniform(0.5, 5.0)  # 500ms-5s
        else:
            execution_time = random.uniform(0.005, 0.5)  # 5ms-500ms
    
    # Determine status code based on anomaly flag
    if is_anomalous and random.random() < 0.8:
        # Higher chance of error for anomalous queries
        status_code = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    else:
        # Normal queries mostly succeed
        weights = [0.95, 0.05]  # 95% success, 5% error or warning
        status_category = random.choices(["success", "error"], weights=weights)[0]
        
        if status_category == "success":
            status_code = 0 if random.random() < 0.9 else 1  # Mostly success, sometimes warning
        else:
            status_code = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Generate error message if applicable
    if status_code > 0:
        if status_code in DB_ERROR_MESSAGES:
            error_message = random.choice(DB_ERROR_MESSAGES[status_code])
            # Format error message if it contains placeholders
            if "{0}" in error_message:
                error_message = error_message.format(int(execution_time * 1000))
        else:
            error_message = "Unknown database error occurred"
    else:
        error_message = None
    
    # Generate affected rows for write operations
    affected_rows = None
    if operation in ["insert", "update", "delete"] and status_code == 0:
        if operation == "insert":
            affected_rows = 1  # Usually inserts affect 1 row
        elif operation == "update" or operation == "delete":
            if "WHERE" in query or ".One(" in query:
                affected_rows = 1
            else:
                # Bulk updates or deletes without WHERE clause
                affected_rows = random.randint(10, 10000)
    
    # Generate rows returned for read operations
    rows_returned = None
    if operation in ["query", "aggregate"] and status_code == 0:
        if "LIMIT" in query:
            # Extract the limit value if present
            try:
                limit = int(query.split("LIMIT")[1].strip().split()[0])
                rows_returned = random.randint(0, min(limit, 1000))
            except:
                rows_returned = random.randint(0, 1000)
        elif ".limit(" in query:
            # Extract the limit value for MongoDB queries
            try:
                limit = int(query.split(".limit(")[1].split(")")[0])
                rows_returned = random.randint(0, min(limit, 1000))
            except:
                rows_returned = random.randint(0, 1000)
        else:
            # No explicit limit
            rows_returned = random.randint(0, 5000)
    
    # Database connection details
    connection_id = f"conn-{uuid.uuid4().hex[:8]}"
    
    # Generate metrics
    metrics = {
        "cpu_time_ms": round(execution_time * random.uniform(0.4, 0.8) * 1000, 2),
        "io_time_ms": round(execution_time * random.uniform(0.2, 0.6) * 1000, 2),
        "memory_usage_kb": random.randint(1000, 50000),
        "lock_time_ms": round(random.uniform(0, execution_time * 0.3) * 1000, 2) if random.random() < 0.3 else 0
    }
    
    # Create the log entry
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "db_type": db_type,
        "instance_id": instance_id,
        "connection_id": connection_id,
        "operation": operation,
        "collection": collection,
        "query": query,
        "execution_time_sec": round(execution_time, 6),
        "status_code": status_code,
        "status_message": DB_STATUS_CODES.get(status_code, "Unknown Status"),
        "error_message": error_message,
        "affected_rows": affected_rows,
        "rows_returned": rows_returned,
        "query_plan": generate_query_plan(db_type, operation, collection, is_anomalous) if random.random() < 0.3 else None,
        "metrics": metrics,
        "correlation_id": correlation_id,
        "is_anomalous": is_anomalous  # Meta field for labeling
    }
    
    return log_entry

def generate_query_plan(db_type, operation, collection, is_anomalous):
    """Generate a simplified query execution plan"""
    if operation not in ["query", "aggregate"]:
        return None
    
    if db_type in ["mysql", "postgres"]:
        # SQL database query plan
        steps = []
        
        if "JOIN" in operation and random.random() < 0.7:
            # For queries with joins
            if is_anomalous and random.random() < 0.6:
                steps.append({
                    "operation": "Nested Loop",
                    "table": collection,
                    "cost": random.uniform(100, 10000),
                    "rows": random.randint(1000, 100000)
                })
            else:
                steps.append({
                    "operation": "Hash Join",
                    "table": collection,
                    "cost": random.uniform(10, 1000),
                    "rows": random.randint(10, 10000)
                })
        
        # Add index scan or table scan
        if is_anomalous and random.random() < 0.7:
            steps.append({
                "operation": "Sequential Scan",
                "table": collection,
                "cost": random.uniform(500, 5000),
                "rows": random.randint(10000, 1000000)
            })
        else:
            steps.append({
                "operation": "Index Scan",
                "table": collection,
                "index": f"idx_{collection}_id",
                "cost": random.uniform(1, 100),
                "rows": random.randint(1, 1000)
            })
        
        return {
            "planner": f"{db_type.capitalize()} Query Planner",
            "steps": steps,
            "total_cost": sum(step["cost"] for step in steps),
            "planning_time_ms": round(random.uniform(0.1, 10.0), 2),
            "execution_time_ms": round(random.uniform(1.0, 1000.0), 2)
        }
    
    elif db_type == "mongodb":
        # MongoDB query plan
        win_stage = {
            "stage": "FETCH" if random.random() < 0.7 else "IXSCAN",
            "inputStage": {
                "stage": "IXSCAN" if random.random() < 0.8 else "COLLSCAN",
                "indexName": f"{collection}_id_idx" if random.random() < 0.8 else None,
                "isMultiKey": False,
                "direction": "forward",
                "indexBounds": "{}"
            } if random.random() < 0.7 else None
        }
        
        return {
            "namespace": f"database.{collection}",
            "parsedQuery": {},
            "winningPlan": win_stage,
            "rejectedPlans": [],
            "executionStats": {
                "executionSuccess": True,
                "nReturned": random.randint(1, 1000),
                "executionTimeMillis": round(random.uniform(1, 1000), 2),
                "totalKeysExamined": random.randint(1, 2000),
                "totalDocsExamined": random.randint(1, 5000),
                "executionStages": {}
            }
        }
    
    return None

def generate_db_flow(base_time=None, correlation_id=None, is_anomalous=False):
    """Generate a sequence of related database operations that form a flow"""
    if base_time is None:
        base_time = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )
    
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Define a database operation flow pattern
    # For example: query (read user) -> query (read product) -> insert (create order) -> update (update inventory)
    flow_pattern = [
        {"operation": "query", "db_type": random.choice(DB_TYPES)},
        {"operation": "query", "db_type": random.choice(DB_TYPES)},
        {"operation": random.choice(["insert", "transaction"]), "db_type": random.choice(DB_TYPES)},
        {"operation": "update", "db_type": random.choice(DB_TYPES)}
    ]
    
    # Determine if we introduce an error and at which step
    error_step = random.randint(0, len(flow_pattern) - 1) if is_anomalous else None
    
    # Generate logs for each step in the flow
    flow_logs = []
    current_time = base_time
    
    for i, step in enumerate(flow_pattern):
        # Add some time between database operations
        current_time += datetime.timedelta(milliseconds=random.randint(5, 100))
        
        # Determine if this step should be anomalous
        step_is_anomalous = is_anomalous and (i == error_step or random.random() < 0.3)
        
        # Generate the log entry
        log_entry = generate_db_log_entry(
            timestamp=current_time,
            db_type=step["db_type"],
            operation=step["operation"],
            correlation_id=correlation_id,
            is_anomalous=step_is_anomalous
        )
        
        flow_logs.append(log_entry)
        
        # If we get an error, we might stop the flow
        if step_is_anomalous and log_entry["status_code"] >= 2 and random.random() < 0.7:
            break
    
    return flow_logs

def generate_db_logs(num_logs=1000, anomaly_percentage=20):
    """Generate a dataset of database API logs with a specified percentage of anomalies"""
    logs = []
    flow_logs = []
    
    # Calculate number of anomalous logs
    num_anomalous = int(num_logs * (anomaly_percentage / 100))
    num_normal = num_logs - num_anomalous
    
    # Generate individual normal logs
    for _ in range(int(num_normal * 0.6)):  # 60% of normal logs are individual
        logs.append(generate_db_log_entry(is_anomalous=False))
    
    # Generate individual anomalous logs
    for _ in range(int(num_anomalous * 0.4)):  # 40% of anomalous logs are individual
        logs.append(generate_db_log_entry(is_anomalous=True))
    
    # Generate normal database flows
    num_normal_flows = int(num_normal * 0.4 / 4)  # Approximately 4 logs per flow
    for _ in range(num_normal_flows):
        flow_logs.extend(generate_db_flow(is_anomalous=False))
    
    # Generate anomalous database flows
    num_anomalous_flows = int(num_anomalous * 0.6 / 4)  # Approximately 4 logs per flow
    for _ in range(num_anomalous_flows):
        flow_logs.extend(generate_db_flow(is_anomalous=True))
    
    # Combine all logs
    all_logs = logs + flow_logs
    
    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(all_logs)
    
    return all_logs

def save_db_logs(logs, format='json', filename='database_api_logs'):
    """Save database logs to a file in the specified format"""
    if format.lower() == 'json':
        file_path = f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved {len(logs)} database logs to {file_path}")
        return file_path
    
    elif format.lower() == 'csv':
        file_path = f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        flat_logs = []
        for log in logs:
            flat_log = {
                "timestamp": log["timestamp"],
                "db_type": log["db_type"],
                "instance_id": log["instance_id"],
                "connection_id": log["connection_id"],
                "operation": log["operation"],
                "collection": log["collection"],
                "query": log["query"],
                "execution_time_sec": log["execution_time_sec"],
                "status_code": log["status_code"],
                "status_message": log["status_message"],
                "error_message": log["error_message"],
                "affected_rows": log["affected_rows"],
                "rows_returned": log["rows_returned"],
                "correlation_id": log["correlation_id"],
                "cpu_time_ms": log["metrics"]["cpu_time_ms"],
                "io_time_ms": log["metrics"]["io_time_ms"],
                "memory_usage_kb": log["metrics"]["memory_usage_kb"],
                "lock_time_ms": log["metrics"]["lock_time_ms"],
                "is_anomalous": log["is_anomalous"]
            }
            flat_logs.append(flat_log)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flat_logs)
        df.to_csv(file_path, index=False)
        print(f"Saved {len(logs)} database logs to {file_path}")
        return file_path
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

def analyze_db_logs(logs):
    """Print analysis of the generated database logs"""
    total_logs = len(logs)
    anomalous_count = sum(1 for log in logs if log.get('is_anomalous', False))
    normal_count = total_logs - anomalous_count
    
    # Count by database type
    db_types = {}
    for log in logs:
        db_type = log['db_type']
        db_types[db_type] = db_types.get(db_type, 0) + 1
    
    # Count by operation
    operations = {}
    for log in logs:
        operation = log['operation']
        operations[operation] = operations.get(operation, 0) + 1
    
    # Count by status code
    status_codes = {}
    for log in logs:
        status = log['status_code']
        status_codes[status] = status_codes.get(status, 0) + 1
    
    # Count unique correlation IDs (database flows)
    correlation_ids = set(log['correlation_id'] for log in logs)
    
    # Calculate average execution times
    normal_exec_times = [log['execution_time_sec'] for log in logs if not log.get('is_anomalous', False)]
    anomalous_exec_times = [log['execution_time_sec'] for log in logs if log.get('is_anomalous', False)]
    
    avg_normal_time = sum(normal_exec_times) / len(normal_exec_times) if normal_exec_times else 0
    avg_anomalous_time = sum(anomalous_exec_times) / len(anomalous_exec_times) if anomalous_exec_times else 0
    
    print("\n=== Database Log Analysis ===")
    print(f"Total logs: {total_logs}")
    print(f"Normal logs: {normal_count} ({normal_count/total_logs*100:.2f}%)")
    print(f"Anomalous logs: {anomalous_count} ({anomalous_count/total_logs*100:.2f}%)")
    print(f"Unique database flows (correlation IDs): {len(correlation_ids)}")
    print(f"Average execution time (normal): {avg_normal_time*1000:.2f} ms")
    print(f"Average execution time (anomalous): {avg_anomalous_time*1000:.2f} ms")
    
    print("\n=== Database Type Distribution ===")
    for db_type, count in sorted(db_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{db_type}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Operation Distribution ===")
    for operation, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
        print(f"{operation}: {count} logs ({count/total_logs*100:.2f}%)")
    
    print("\n=== Status Code Distribution ===")
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        print(f"Status {code} ({DB_STATUS_CODES.get(code, 'Unknown')}): {count} logs ({count/total_logs*100:.2f}%)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate 1000 logs with 20% anomalies
    logs = generate_db_logs(num_logs=1000, anomaly_percentage=20)
    
    # Analyze the logs
    analyze_db_logs(logs)
    
    # Save logs to JSON and CSV
    json_path = save_db_logs(logs, format='json')
    csv_path = save_db_logs(logs, format='csv')
    
    print(f"\nDatabase logs have been saved to {json_path} and {csv_path}")
    print("You can use these files for training your anomaly detection model.")
    
    # Print sample logs (1 normal, 1 anomalous)
    print("\n=== Sample Normal Database Log ===")
    normal_log = next(log for log in logs if not log.get('is_anomalous', False))
    print(json.dumps(normal_log, indent=2))
    
    print("\n=== Sample Anomalous Database Log ===")
    anomalous_log = next(log for log in logs if log.get('is_anomalous', True))
    print(json.dumps(anomalous_log, indent=2))