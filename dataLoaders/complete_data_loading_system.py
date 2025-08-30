#!/usr/bin/env python3
"""
Complete Data Loading System for All 4 Databases
Loads the full 12.4M Chicago 311 records into MongoDB, Elasticsearch, PostgreSQL, and DuckDB
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import requests
import pandas as pd
import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteDataLoader:
    """Complete data loader for all 4 databases."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 5000
        self.max_retries = 3
        self.total_records = 0
        self.loaded_records = {}
        self.databases = {}
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        
    def setup_database_connections(self):
        """Setup connections to all 4 databases."""
        print("🔄 Setting up all 4 database connections...")
        
        # 1. MongoDB
        try:
            mongo_handler = MongoDBHandler()
            # Clear existing data to start fresh
            mongo_handler.collection.delete_many({})
            self.databases['MongoDB'] = mongo_handler
            self.loaded_records['MongoDB'] = 0
            print("✅ MongoDB connected and cleared")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # 2. Elasticsearch  
        try:
            es_handler = ElasticsearchHandler()
            # Clear existing data
            try:
                es_handler.es.delete_by_query(
                    index=es_handler.index_name,
                    body={"query": {"match_all": {}}}
                )
            except:
                pass
            self.databases['Elasticsearch'] = es_handler
            self.loaded_records['Elasticsearch'] = 0
            print("✅ Elasticsearch connected and cleared")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        # 3. Local PostgreSQL
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # Create PostgreSQL handler
            class SimplePostgreSQLHandler:
                def __init__(self):
                    self.connection = None
                    self.setup_and_connect()
                
                def setup_and_connect(self):
                    try:
                        # Try to connect to default postgres database first
                        conn = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="postgres",
                            user="postgres",
                            password=""
                        )
                        conn.autocommit = True
                        cursor = conn.cursor()
                        
                        # Drop and recreate database for fresh start
                        cursor.execute("DROP DATABASE IF EXISTS chicago_311_full")
                        cursor.execute("CREATE DATABASE chicago_311_full")
                        
                        cursor.close()
                        conn.close()
                        
                        # Now connect to our database
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="chicago_311_full",
                            user="postgres",
                            password="",
                            cursor_factory=RealDictCursor
                        )
                        
                        # Create table
                        self.create_table()
                        print("✅ PostgreSQL connected and table created")
                        
                    except Exception as e:
                        print(f"❌ PostgreSQL setup failed: {e}")
                        raise
                
                def create_table(self):
                    create_sql = """
                    CREATE TABLE IF NOT EXISTS chicago_311_requests (
                        sr_number VARCHAR(50) PRIMARY KEY,
                        sr_type VARCHAR(200),
                        sr_short_code VARCHAR(10),
                        owner_department VARCHAR(100),
                        status VARCHAR(50),
                        origin VARCHAR(50),
                        created_date TIMESTAMP,
                        last_modified_date TIMESTAMP,
                        closed_date TIMESTAMP,
                        street_address VARCHAR(200),
                        city VARCHAR(50),
                        state VARCHAR(50),
                        zip_code VARCHAR(20),
                        street_number VARCHAR(20),
                        street_direction VARCHAR(10),
                        street_name VARCHAR(100),
                        street_type VARCHAR(20),
                        duplicate BOOLEAN DEFAULT FALSE,
                        legacy_record BOOLEAN DEFAULT FALSE,
                        latitude DECIMAL(10, 7),
                        longitude DECIMAL(10, 7),
                        ward INTEGER,
                        community_area INTEGER,
                        police_district VARCHAR(20)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_sr_type ON chicago_311_requests(sr_type);
                    CREATE INDEX IF NOT EXISTS idx_status ON chicago_311_requests(status);
                    CREATE INDEX IF NOT EXISTS idx_created_date ON chicago_311_requests(created_date);
                    CREATE INDEX IF NOT EXISTS idx_ward ON chicago_311_requests(ward);
                    """
                    
                    cursor = self.connection.cursor()
                    cursor.execute(create_sql)
                    self.connection.commit()
                    cursor.close()
                
                def insert_batch(self, batch_data):
                    cursor = self.connection.cursor()
                    try:
                        # Prepare data for insertion
                        columns = list(batch_data[0].keys())
                        placeholders = ', '.join(['%s'] * len(columns))
                        insert_sql = f"""
                            INSERT INTO chicago_311_requests ({', '.join(columns)}) 
                            VALUES ({placeholders})
                            ON CONFLICT (sr_number) DO NOTHING
                        """
                        
                        values = []
                        for record in batch_data:
                            values.append([record.get(col) for col in columns])
                        
                        cursor.executemany(insert_sql, values)
                        self.connection.commit()
                        return cursor.rowcount
                    except Exception as e:
                        self.connection.rollback()
                        logger.error(f"PostgreSQL insert failed: {e}")
                        return 0
                    finally:
                        cursor.close()
                
                def count_records(self):
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chicago_311_requests")
                    count = cursor.fetchone()[0]
                    cursor.close()
                    return count
                
                def close(self):
                    if self.connection:
                        self.connection.close()
            
            pg_handler = SimplePostgreSQLHandler()
            self.databases['PostgreSQL'] = pg_handler
            self.loaded_records['PostgreSQL'] = 0
            print("✅ PostgreSQL connected with fresh database")
            
        except Exception as e:
            print(f"❌ PostgreSQL failed: {e}")
            print("   Install PostgreSQL: brew install postgresql && brew services start postgresql")
        
        # 4. DuckDB
        try:
            import duckdb
            
            class SimpleDuckDBHandler:
                def __init__(self):
                    self.db_path = "chicago_311_full.duckdb"
                    # Remove existing file for fresh start
                    if os.path.exists(self.db_path):
                        os.remove(self.db_path)
                    
                    self.connection = duckdb.connect(self.db_path)
                    self.create_table()
                    print("✅ DuckDB connected with fresh database")
                
                def create_table(self):
                    create_sql = """
                    CREATE TABLE chicago_311_requests (
                        sr_number VARCHAR PRIMARY KEY,
                        sr_type VARCHAR,
                        sr_short_code VARCHAR,
                        owner_department VARCHAR,
                        status VARCHAR,
                        origin VARCHAR,
                        created_date TIMESTAMP,
                        last_modified_date TIMESTAMP,
                        closed_date TIMESTAMP,
                        street_address VARCHAR,
                        city VARCHAR,
                        state VARCHAR,
                        zip_code VARCHAR,
                        street_number VARCHAR,
                        street_direction VARCHAR,
                        street_name VARCHAR,
                        street_type VARCHAR,
                        duplicate BOOLEAN,
                        legacy_record BOOLEAN,
                        latitude DECIMAL,
                        longitude DECIMAL,
                        ward INTEGER,
                        community_area INTEGER,
                        police_district VARCHAR
                    );
                    """
                    
                    self.connection.execute(create_sql)
                
                def insert_batch(self, batch_data):
                    try:
                        df = pd.DataFrame(batch_data)
                        self.connection.register('temp_df', df)
                        self.connection.execute("""
                            INSERT OR IGNORE INTO chicago_311_requests 
                            SELECT * FROM temp_df
                        """)
                        self.connection.unregister('temp_df')
                        return len(batch_data)
                    except Exception as e:
                        logger.error(f"DuckDB insert failed: {e}")
                        return 0
                
                def count_records(self):
                    result = self.connection.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
                    return result[0] if result else 0
                
                def close(self):
                    if self.connection:
                        self.connection.close()
            
            duckdb_handler = SimpleDuckDBHandler()
            self.databases['DuckDB'] = duckdb_handler
            self.loaded_records['DuckDB'] = 0
            print("✅ DuckDB connected with fresh database")
            
        except Exception as e:
            print(f"❌ DuckDB failed: {e}")
        
        if not self.databases:
            raise Exception("No databases available!")
        
        print(f"📊 Ready to load data into: {list(self.databases.keys())}")
    
    def fetch_data_batch(self, offset: int, limit: int) -> List[Dict]:
        """Fetch a batch of data from Chicago 311 API."""
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': 'sr_number'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for offset {offset}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return []
    
    def clean_record(self, record: Dict) -> Dict:
        """Clean and standardize a record for database insertion."""
        cleaned = {}
        
        # Basic fields
        basic_fields = [
            'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 
            'status', 'origin', 'street_address', 'city', 'state', 'zip_code',
            'street_number', 'street_direction', 'street_name', 'street_type'
        ]
        
        for field in basic_fields:
            value = record.get(field)
            cleaned[field] = str(value).strip() if value else None
        
        # Numeric fields
        for field in ['ward', 'community_area']:
            value = record.get(field)
            try:
                cleaned[field] = int(float(value)) if value else None
            except (ValueError, TypeError):
                cleaned[field] = None
        
        # Decimal fields
        for field in ['latitude', 'longitude']:
            value = record.get(field)
            try:
                cleaned[field] = float(value) if value else None
            except (ValueError, TypeError):
                cleaned[field] = None
        
        # Date fields
        for field in ['created_date', 'last_modified_date', 'closed_date']:
            value = record.get(field)
            if value:
                try:
                    cleaned[field] = datetime.fromisoformat(value.replace('T', ' ').replace('Z', ''))
                except:
                    cleaned[field] = None
            else:
                cleaned[field] = None
        
        # Boolean fields
        cleaned['duplicate'] = str(record.get('duplicate', '')).lower() == 'true'
        cleaned['legacy_record'] = str(record.get('legacy_record', '')).lower() == 'true'
        cleaned['police_district'] = record.get('police_district')
        
        return cleaned
    
    def load_batch_to_database(self, db_name: str, batch_data: List[Dict]) -> int:
        """Load a batch of data to a specific database."""
        if db_name not in self.databases:
            return 0
        
        try:
            handler = self.databases[db_name]
            
            if db_name == 'MongoDB':
                result = handler.collection.insert_many(batch_data, ordered=False)
                return len(result.inserted_ids)
            
            elif db_name == 'Elasticsearch':
                from elasticsearch.helpers import bulk
                actions = []
                for doc in batch_data:
                    # Convert datetime to string for ES
                    es_doc = {}
                    for key, value in doc.items():
                        if isinstance(value, datetime):
                            es_doc[key] = value.isoformat()
                        else:
                            es_doc[key] = value
                    
                    actions.append({
                        "_index": handler.index_name,
                        "_id": doc['sr_number'],
                        "_source": es_doc
                    })
                
                success_count, _ = bulk(handler.es, actions, request_timeout=60)
                return success_count
            
            elif db_name in ['PostgreSQL', 'DuckDB']:
                return handler.insert_batch(batch_data)
            
        except Exception as e:
            if "duplicate key" not in str(e).lower():
                logger.error(f"{db_name} batch insert failed: {e}")
            return 0
    
    def load_all_data(self, max_records: int = None):
        """Load data into all databases."""
        print("🚀 Starting complete data loading for all databases...")
        
        # Estimate total records
        try:
            response = requests.get(f"{self.api_url}?$select=count(*)", timeout=30)
            if response.status_code == 200:
                total_records = int(response.json()[0]['count'])
            else:
                total_records = 12_400_000
        except:
            total_records = 12_400_000
        
        if max_records:
            total_records = min(total_records, max_records)
        
        print(f"📊 Target records: {total_records:,}")
        self.total_records = total_records
        
        # Progress tracking
        processed_records = 0
        start_time = time.time()
        last_progress_time = start_time
        
        # Load data in batches
        offset = 0
        consecutive_failures = 0
        max_failures = 5
        
        while offset < total_records and consecutive_failures < max_failures:
            try:
                current_batch_size = min(self.batch_size, total_records - offset)
                
                print(f"📥 Fetching batch at offset {offset:,}")
                batch_data = self.fetch_data_batch(offset, current_batch_size)
                
                if not batch_data:
                    print("⚠️ No more data available")
                    break
                
                # Clean data
                cleaned_batch = [self.clean_record(record) for record in batch_data]
                
                # Load to all databases in parallel
                batch_results = {}
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    
                    for db_name in self.databases.keys():
                        futures[db_name] = executor.submit(
                            self.load_batch_to_database, db_name, cleaned_batch
                        )
                    
                    # Collect results
                    for db_name, future in futures.items():
                        try:
                            loaded_count = future.result(timeout=180)
                            batch_results[db_name] = loaded_count
                            self.loaded_records[db_name] += loaded_count
                        except Exception as e:
                            batch_results[db_name] = 0
                            logger.error(f"{db_name} loading failed: {e}")
                
                # Report batch results
                for db_name, count in batch_results.items():
                    print(f"   ✅ {db_name}: +{count} records")
                
                processed_records += len(batch_data)
                offset += len(batch_data)
                consecutive_failures = 0
                
                # Progress report every 50k records
                if processed_records % 50000 == 0 or time.time() - last_progress_time > 120:
                    elapsed = time.time() - start_time
                    progress = (processed_records / total_records) * 100
                    rate = processed_records / elapsed if elapsed > 0 else 0
                    eta = (total_records - processed_records) / rate if rate > 0 else 0
                    
                    print(f"\n📊 Progress: {progress:.1f}% ({processed_records:,}/{total_records:,})")
                    print(f"⏱️ Rate: {rate:.0f} records/sec, ETA: {eta/60:.0f} min")
                    print("📈 Total loaded by database:")
                    for db, count in self.loaded_records.items():
                        print(f"   {db}: {count:,}")
                    print()
                    
                    last_progress_time = time.time()
                
                # Small delay
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Batch failed at offset {offset}: {e}")
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    print(f"⏸️ Waiting before retry... ({consecutive_failures}/{max_failures})")
                    time.sleep(30)
                else:
                    print("💥 Too many failures. Stopping.")
                    break
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 DATA LOADING COMPLETED")
        print("="*80)
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Records processed: {processed_records:,}")
        print(f"📈 Average rate: {processed_records/total_time:.0f} records/second")
    
    def verify_all_databases(self):
        """Verify data in all databases."""
        print("\n🔍 Verifying data in all databases...")
        
        verification_results = {}
        
        for db_name, handler in self.databases.items():
            try:
                if db_name == 'MongoDB':
                    count = handler.collection.count_documents({})
                elif db_name == 'Elasticsearch':
                    result = handler.es.count(index=handler.index_name)
                    count = result['count']
                elif db_name in ['PostgreSQL', 'DuckDB']:
                    count = handler.count_records()
                
                verification_results[db_name] = count
                print(f"✅ {db_name}: {count:,} records verified")
                
            except Exception as e:
                print(f"❌ {db_name} verification failed: {e}")
                verification_results[db_name] = -1
        
        return verification_results
    
    def close_connections(self):
        """Close all connections."""
        for handler in self.databases.values():
            try:
                handler.close()
            except:
                pass

def main():
    """Main function to load complete dataset."""
    loader = CompleteDataLoader()
    
    try:
        loader.setup_database_connections()
        
        print("\nData Loading Options:")
        print("1. Load 100K records (quick test)")
        print("2. Load 1M records (medium test)")  
        print("3. Load ALL 12.4M records (full dataset)")
        
        choice = input("Enter choice (1-3) [default: 1]: ").strip()
        
        if choice == '2':
            max_records = 1_000_000
        elif choice == '3':
            max_records = None  # Load all
        else:
            max_records = 100_000
        
        print(f"\n🎯 Starting data load (target: {max_records or '12.4M'} records)")
        
        # Load data
        loader.load_all_data(max_records)
        
        # Verify
        verification_results = loader.verify_all_databases()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'target_records': max_records or loader.total_records,
            'loaded_records': loader.loaded_records,
            'verification_results': verification_results
        }
        
        with open('complete_data_loading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to 'complete_data_loading_results.json'")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close_connections()

if __name__ == "__main__":
    main()