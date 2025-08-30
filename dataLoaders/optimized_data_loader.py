#!/usr/bin/env python3
"""
Optimized Chicago 311 Data Loader
Efficiently loads data into available databases with local DB creation
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
from src.databases.duckdb_handler import DuckDBHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDataLoader:
    """Optimized data loader for Chicago 311 data."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 5000
        self.max_retries = 3
        self.total_records = 0
        self.loaded_records = {}
        self.databases = {}
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.progress_queue = queue.Queue()
        
    def setup_database_connections(self):
        """Setup connections to all available databases."""
        print("🔄 Setting up database connections...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            self.databases['MongoDB'] = mongo_handler
            self.loaded_records['MongoDB'] = 0
            print("✅ MongoDB connected")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # Elasticsearch  
        try:
            es_handler = ElasticsearchHandler()
            self.databases['Elasticsearch'] = es_handler
            self.loaded_records['Elasticsearch'] = 0
            print("✅ Elasticsearch connected")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        # Local PostgreSQL (try to create if not exists)
        try:
            from src.databases.local_postgresql_handler import LocalPostgreSQLHandler
            pg_handler = LocalPostgreSQLHandler()
            self.create_postgresql_table(pg_handler)
            self.databases['PostgreSQL'] = pg_handler
            self.loaded_records['PostgreSQL'] = 0
            print("✅ Local PostgreSQL connected and table created")
        except Exception as e:
            print(f"❌ Local PostgreSQL failed: {e}")
            print("   Consider installing PostgreSQL: brew install postgresql")
        
        # DuckDB - always create local file
        try:
            # Create local DuckDB file
            duckdb_file = "chicago_311_local.duckdb"
            duckdb_handler = DuckDBHandler(db_path=duckdb_file)
            self.create_duckdb_table(duckdb_handler)
            self.databases['DuckDB'] = duckdb_handler
            self.loaded_records['DuckDB'] = 0
            print("✅ DuckDB connected and table created")
        except Exception as e:
            print(f"❌ DuckDB failed: {e}")
        
        if not self.databases:
            raise Exception("No databases available!")
        
        print(f"📊 Connected databases: {list(self.databases.keys())}")
    
    def create_postgresql_table(self, handler):
        """Create PostgreSQL table for Chicago 311 data."""
        create_table_sql = """
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
        
        with handler.get_cursor() as cursor:
            cursor.execute(create_table_sql)
        print("✅ PostgreSQL table and indexes created")
    
    def create_duckdb_table(self, handler):
        """Create DuckDB table for Chicago 311 data."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chicago_311_requests (
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
        
        handler.connection.execute(create_table_sql)
        print("✅ DuckDB table created")
    
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
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def clean_record(self, record: Dict) -> Dict:
        """Clean and standardize a record for database insertion."""
        cleaned = {}
        
        # Basic field extraction
        basic_fields = [
            'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 
            'status', 'origin', 'street_address', 'city', 'state', 'zip_code',
            'street_number', 'street_direction', 'street_name', 'street_type'
        ]
        
        for field in basic_fields:
            value = record.get(field)
            cleaned[field] = str(value).strip() if value else None
        
        # Handle numeric fields
        numeric_fields = ['ward', 'community_area']
        for field in numeric_fields:
            value = record.get(field)
            try:
                cleaned[field] = int(float(value)) if value else None
            except (ValueError, TypeError):
                cleaned[field] = None
        
        # Handle decimal fields
        decimal_fields = ['latitude', 'longitude']
        for field in decimal_fields:
            value = record.get(field)
            try:
                cleaned[field] = float(value) if value else None
            except (ValueError, TypeError):
                cleaned[field] = None
        
        # Handle date fields
        date_fields = ['created_date', 'last_modified_date', 'closed_date']
        for field in date_fields:
            value = record.get(field)
            if value:
                try:
                    # Parse ISO format dates
                    cleaned[field] = datetime.fromisoformat(value.replace('T', ' ').replace('Z', ''))
                except:
                    cleaned[field] = None
            else:
                cleaned[field] = None
        
        # Handle boolean fields
        cleaned['duplicate'] = str(record.get('duplicate', '')).lower() == 'true'
        cleaned['legacy_record'] = str(record.get('legacy_record', '')).lower() == 'true'
        
        # Handle other fields
        cleaned['police_district'] = record.get('police_district')
        
        return cleaned
    
    def load_batch_to_mongodb(self, batch_data: List[Dict]) -> int:
        """Load a batch of data to MongoDB."""
        if 'MongoDB' not in self.databases:
            return 0
        
        try:
            handler = self.databases['MongoDB']
            # Use insert_many with ordered=False for better performance
            result = handler.collection.insert_many(batch_data, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            # Handle duplicate key errors
            if "duplicate key" in str(e).lower():
                # Try inserting one by one to handle duplicates
                count = 0
                for doc in batch_data:
                    try:
                        handler.collection.insert_one(doc)
                        count += 1
                    except:
                        pass  # Skip duplicates
                return count
            else:
                logger.error(f"MongoDB batch insert failed: {e}")
                return 0
    
    def load_batch_to_elasticsearch(self, batch_data: List[Dict]) -> int:
        """Load a batch of data to Elasticsearch."""
        if 'Elasticsearch' not in self.databases:
            return 0
        
        try:
            from elasticsearch.helpers import bulk
            handler = self.databases['Elasticsearch']
            
            actions = []
            for doc in batch_data:
                # Convert datetime objects to strings for ES
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
        except Exception as e:
            logger.error(f"Elasticsearch batch insert failed: {e}")
            return 0
    
    def load_batch_to_postgresql(self, batch_data: List[Dict]) -> int:
        """Load a batch of data to PostgreSQL."""
        if 'PostgreSQL' not in self.databases:
            return 0
        
        try:
            handler = self.databases['PostgreSQL']
            
            # Prepare INSERT statement
            columns = list(batch_data[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            insert_sql = f"""
                INSERT INTO chicago_311_requests ({', '.join(columns)}) 
                VALUES ({placeholders})
                ON CONFLICT (sr_number) DO NOTHING
            """
            
            with handler.get_cursor() as cursor:
                values = []
                for record in batch_data:
                    values.append([record.get(col) for col in columns])
                
                cursor.executemany(insert_sql, values)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"PostgreSQL batch insert failed: {e}")
            return 0
    
    def load_batch_to_duckdb(self, batch_data: List[Dict]) -> int:
        """Load a batch of data to DuckDB."""
        if 'DuckDB' not in self.databases:
            return 0
        
        try:
            handler = self.databases['DuckDB']
            
            # Convert to DataFrame for easier insertion
            df = pd.DataFrame(batch_data)
            
            # Use DuckDB's efficient DataFrame insertion with upsert
            handler.connection.register('temp_df', df)
            handler.connection.execute("""
                INSERT OR IGNORE INTO chicago_311_requests 
                SELECT * FROM temp_df
            """)
            handler.connection.unregister('temp_df')
            
            return len(batch_data)
        except Exception as e:
            logger.error(f"DuckDB batch insert failed: {e}")
            return 0
    
    def load_data_progressive(self, max_records: int = None):
        """Load data progressively with smart batching."""
        print("🚀 Starting progressive data loading...")
        
        # Get estimated total records
        print("📊 Checking total available records...")
        try:
            # Try to get exact count (this might not work on all endpoints)
            count_params = {'$select': 'count(*)'}
            response = requests.get(self.api_url, params=count_params, timeout=30)
            if response.status_code == 200:
                total_records = int(response.json()[0]['count'])
            else:
                # Fallback: estimate based on known data size
                total_records = 12_400_000
        except:
            total_records = 12_400_000
        
        if max_records:
            total_records = min(total_records, max_records)
        
        print(f"📊 Target records to load: {total_records:,}")
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
                
                # Fetch batch
                print(f"📥 Fetching batch at offset {offset:,} (size: {current_batch_size})")
                batch_data = self.fetch_data_batch(offset, current_batch_size)
                
                if not batch_data:
                    print("⚠️ No more data available")
                    break
                
                # Clean the data
                cleaned_batch = [self.clean_record(record) for record in batch_data]
                
                # Load to all databases in parallel
                batch_results = {}
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    
                    if 'MongoDB' in self.databases:
                        futures['MongoDB'] = executor.submit(self.load_batch_to_mongodb, cleaned_batch)
                    
                    if 'Elasticsearch' in self.databases:
                        futures['Elasticsearch'] = executor.submit(self.load_batch_to_elasticsearch, cleaned_batch)
                    
                    if 'PostgreSQL' in self.databases:
                        futures['PostgreSQL'] = executor.submit(self.load_batch_to_postgresql, cleaned_batch)
                    
                    if 'DuckDB' in self.databases:
                        futures['DuckDB'] = executor.submit(self.load_batch_to_duckdb, cleaned_batch)
                    
                    # Collect results
                    for db_name, future in futures.items():
                        try:
                            loaded_count = future.result(timeout=120)
                            batch_results[db_name] = loaded_count
                            self.loaded_records[db_name] += loaded_count
                        except Exception as e:
                            print(f"   ❌ {db_name}: {e}")
                            batch_results[db_name] = 0
                
                # Report batch results
                for db_name, count in batch_results.items():
                    print(f"   ✅ {db_name}: {count} records")
                
                processed_records += len(batch_data)
                offset += len(batch_data)
                consecutive_failures = 0
                
                # Progress report every 50,000 records
                if processed_records % 50000 == 0 or time.time() - last_progress_time > 60:
                    elapsed = time.time() - start_time
                    progress = (processed_records / total_records) * 100
                    rate = processed_records / elapsed if elapsed > 0 else 0
                    eta = (total_records - processed_records) / rate if rate > 0 else 0
                    
                    print(f"\n📊 Progress: {progress:.1f}% ({processed_records:,}/{total_records:,})")
                    print(f"⏱️ Elapsed: {elapsed/60:.1f}min, Rate: {rate:.0f} records/sec")
                    print(f"🕒 ETA: {eta/60:.1f} minutes remaining")
                    print("📈 Records loaded by database:")
                    for db, count in self.loaded_records.items():
                        print(f"   {db}: {count:,}")
                    print()
                    
                    last_progress_time = time.time()
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Batch loading failed at offset {offset}: {e}")
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    print(f"⏸️ Waiting 30 seconds before retry... ({consecutive_failures}/{max_failures})")
                    time.sleep(30)
                else:
                    print("💥 Too many consecutive failures. Stopping.")
                    break
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 DATA LOADING COMPLETED")
        print("="*80)
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Total records processed: {processed_records:,}")
        print(f"📈 Average rate: {processed_records/total_time:.0f} records/second")
        print("\n📈 Final record counts:")
        for db_name, count in self.loaded_records.items():
            print(f"   {db_name}: {count:,} records")
    
    def verify_data_loading(self):
        """Verify data was loaded correctly in all databases."""
        print("\n🔍 Verifying data loading...")
        
        verification_results = {}
        
        for db_name, handler in self.databases.items():
            try:
                if db_name == 'MongoDB':
                    count = handler.collection.count_documents({})
                elif db_name == 'Elasticsearch':
                    result = handler.es.count(index=handler.index_name)
                    count = result['count']
                elif db_name == 'PostgreSQL':
                    result, _ = handler.execute_query("SELECT COUNT(*) FROM chicago_311_requests")
                    count = result[0][0]
                elif db_name == 'DuckDB':
                    result, _ = handler.execute_query("SELECT COUNT(*) FROM chicago_311_requests")
                    count = result[0][0]
                
                verification_results[db_name] = count
                print(f"✅ {db_name}: {count:,} records verified")
                
                # Get sample record to verify structure
                if count > 0:
                    if db_name == 'MongoDB':
                        sample = handler.collection.find_one()
                        print(f"   Sample fields: {list(sample.keys())[:8]}...")
                    elif db_name == 'Elasticsearch':
                        result = handler.es.search(index=handler.index_name, size=1)
                        if result['hits']['hits']:
                            sample = result['hits']['hits'][0]['_source']
                            print(f"   Sample fields: {list(sample.keys())[:8]}...")
                    elif db_name in ['PostgreSQL', 'DuckDB']:
                        result, _ = handler.execute_query("SELECT * FROM chicago_311_requests LIMIT 1")
                        if result:
                            print(f"   Sample fields: {list(result[0].keys())[:8]}...")
                
            except Exception as e:
                print(f"❌ {db_name} verification failed: {e}")
                verification_results[db_name] = -1
        
        return verification_results
    
    def close_connections(self):
        """Close all database connections."""
        for handler in self.databases.values():
            try:
                handler.close()
            except:
                pass

def main():
    """Main function to load Chicago 311 data."""
    loader = OptimizedDataLoader()
    
    try:
        loader.setup_database_connections()
        
        if not loader.databases:
            print("❌ No databases available. Exiting.")
            return
        
        # Ask user for loading preference
        print("\nChoose loading option:")
        print("1. Load 50,000 records (test)")
        print("2. Load 500,000 records (medium test)")
        print("3. Load all available records (~12.4M)")
        
        choice = input("Enter choice (1-3) [default: 1]: ").strip()
        
        if choice == '2':
            max_records = 500000
        elif choice == '3':
            max_records = None  # Load all
        else:
            max_records = 50000  # Default test size
        
        print(f"\n🎯 Starting data load (max records: {max_records or 'unlimited'})")
        
        # Load the data
        loader.load_data_progressive(max_records)
        
        # Verify the loading
        verification_results = loader.verify_data_loading()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'loaded_records': loader.loaded_records,
            'verification_results': verification_results,
            'target_records': max_records or loader.total_records
        }
        
        with open('optimized_loading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Data loading results saved to 'optimized_loading_results.json'")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close_connections()

if __name__ == "__main__":
    main()