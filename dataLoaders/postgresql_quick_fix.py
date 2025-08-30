import json
import psycopg2

def serialize_for_postgres(value):
    """Convert dict values to JSON strings for PostgreSQL."""
    if isinstance(value, dict):
        return json.dumps(value)
    elif isinstance(value, list):
        return json.dumps(value) 
    elif value is None or str(value) in ['', 'nan', 'None']:
        return None
    else:
        return str(value)[:500] if isinstance(value, str) and len(value) > 500 else value

print("PostgreSQL serialization fix ready!")