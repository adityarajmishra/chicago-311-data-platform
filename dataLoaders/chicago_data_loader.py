#!/usr/bin/env python3
"""
Chicago 311 Data Loader
Downloads and loads the complete 12.4M Chicago 311 dataset into all databases
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import requests
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler
from src.databases.postgresql_handler import PostgreSQLHandler
from src.databases.duckdb_handler import DuckDBHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chicago311DataLoader:
    """Complete Chicago 311 data loading system."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 10000
        self.max_retries = 3
        self.total_records = 0
        self.loaded_records = {
            'MongoDB': 0,
            'Elasticsearch': 0,
            'PostgreSQL': 0,
            'DuckDB': 0
        }
        
        # Database connections
        self.databases = {}
        
    def setup_database_connections(self):
        """Setup connections to all databases."""
        print("🔄 Setting up database connections...")
        
        # MongoDB
        try:
            mongo_handler = MongoDBHandler()
            self.databases['MongoDB'] = mongo_handler
            print("✅ MongoDB connected")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # Elasticsearch  
        try:
            es_handler = ElasticsearchHandler()
            self.databases['Elasticsearch'] = es_handler
            print("✅ Elasticsearch connected")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        # PostgreSQL
        try:
            pg_handler = PostgreSQLHandler()
            # Create table if it doesn't exist
            self.create_postgresql_table(pg_handler)
            self.databases['PostgreSQL'] = pg_handler
            print("✅ PostgreSQL connected and table created")
        except Exception as e:
            print(f"❌ PostgreSQL failed: {e}")
        
        # DuckDB
        try:
            duckdb_handler = DuckDBHandler()
            # Create table if it doesn't exist
            self.create_duckdb_table(duckdb_handler)
            self.databases['DuckDB'] = duckdb_handler
            print("✅ DuckDB connected and table created")
        except Exception as e:
            print(f"❌ DuckDB failed: {e}")
    
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
            service_request_number VARCHAR(50),
            type_of_service_request VARCHAR(200),
            street_address VARCHAR(200),
            city VARCHAR(50),
            state VARCHAR(10),
            zip_code VARCHAR(20),
            street_number VARCHAR(20),
            street_direction VARCHAR(10),
            street_name VARCHAR(100),
            street_type VARCHAR(20),
            duplicate BOOLEAN,
            legacy_record BOOLEAN,
            legacy_sr_number VARCHAR(50),
            parent_sr_number VARCHAR(50),
            community_area INTEGER,
            ward INTEGER,
            electrical_district VARCHAR(20),
            electricity_grid VARCHAR(20),
            police_district VARCHAR(20),
            latitude DECIMAL(10, 7),
            longitude DECIMAL(10, 7),
            location VARCHAR(100),
            historical_wards_03_15 INTEGER,
            zip_codes INTEGER,
            community_areas INTEGER,
            census_tracts INTEGER,
            wards INTEGER
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
            service_request_number VARCHAR,
            type_of_service_request VARCHAR,
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
            legacy_sr_number VARCHAR,
            parent_sr_number VARCHAR,
            community_area INTEGER,
            ward INTEGER,
            electrical_district VARCHAR,
            electricity_grid VARCHAR,
            police_district VARCHAR,
            latitude DECIMAL,
            longitude DECIMAL,
            location VARCHAR,
            historical_wards_03_15 INTEGER,
            zip_codes INTEGER,
            community_areas INTEGER,
            census_tracts INTEGER,
            wards INTEGER
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
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def clean_record(self, record: Dict) -> Dict:
        """Clean and standardize a record for database insertion."""
        cleaned = {}
        
        # Handle common field mappings and cleaning
        field_mapping = {
            'sr_number': 'sr_number',
            'sr_type': 'sr_type', 
            'sr_short_code': 'sr_short_code',
            'owner_department': 'owner_department',
            'status': 'status',
            'origin': 'origin',
            'created_date': 'created_date',
            'last_modified_date': 'last_modified_date',
            'closed_date': 'closed_date',
            'street_address': 'street_address',
            'city': 'city',
            'state': 'state',
            'zip_code': 'zip_code',
            'ward': 'ward',
            'community_area': 'community_area',
            'police_district': 'police_district',
            'latitude': 'latitude',
            'longitude': 'longitude'
        }
        
        for api_field, db_field in field_mapping.items():
            value = record.get(api_field)
            
            # Clean and convert values
            if value is None or value == '':
                cleaned[db_field] = None
            elif db_field in ['ward', 'community_area']:
                try:
                    cleaned[db_field] = int(float(value)) if value else None
                except (ValueError, TypeError):
                    cleaned[db_field] = None
            elif db_field in ['latitude', 'longitude']:
                try:
                    cleaned[db_field] = float(value) if value else None
                except (ValueError, TypeError):
                    cleaned[db_field] = None
            elif db_field in ['created_date', 'last_modified_date', 'closed_date']:
                # Handle date formatting
                if value:
                    try:
                        # Parse ISO format dates
                        cleaned[db_field] = datetime.fromisoformat(value.replace('T', ' ').replace('Z', ''))
                    except:
                        cleaned[db_field] = None
                else:
                    cleaned[db_field] = None
            else:
                cleaned[db_field] = str(value) if value else None
        
        # Add computed fields
        if cleaned.get('latitude') and cleaned.get('longitude'):
            cleaned['location'] = f"({cleaned['latitude']}, {cleaned['longitude']})"
        
        # Handle boolean fields
        cleaned['duplicate'] = record.get('duplicate', '').lower() == 'true'
        cleaned['legacy_record'] = record.get('legacy_record', '').lower() == 'true'
        
        return cleaned
    
    def load_batch_to_mongodb(self, batch_data: List[Dict]) -> int:
        """Load a batch of data to MongoDB."""
        if 'MongoDB' not in self.databases:
            return 0
        
        try:
            handler = self.databases['MongoDB']
            result = handler.collection.insert_many(batch_data, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
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
            
            success_count, _ = bulk(handler.es, actions)
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
            
            # Insert using DuckDB's efficient DataFrame insertion
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
    
    def load_data_parallel(self):
        """Load data in parallel to all databases."""
        print("🚀 Starting parallel data loading...")
        
        # First, get total record count
        try:
            response = requests.get(f"{self.api_url}?$select=count(*)")
            total_records = int(response.json()[0]['count'])
            print(f"📊 Total records to load: {total_records:,}")
        except:
            total_records = 12_400_000  # Fallback estimate
            print(f"📊 Estimated records to load: {total_records:,}")
        
        self.total_records = total_records
        
        # Progress tracking
        processed_batches = 0
        total_batches = (total_records + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for offset in range(0, total_records, self.batch_size):
            try:
                # Fetch batch
                print(f"📥 Fetching batch {processed_batches + 1}/{total_batches} (offset: {offset:,})")
                batch_data = self.fetch_data_batch(offset, self.batch_size)
                
                if not batch_data:
                    print("⚠️ No more data available")
                    break
                
                # Clean the data
                cleaned_batch = [self.clean_record(record) for record in batch_data]
                
                # Load to all databases in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    
                    if 'MongoDB' in self.databases:
                        futures.append(('MongoDB', executor.submit(self.load_batch_to_mongodb, cleaned_batch)))
                    
                    if 'Elasticsearch' in self.databases:
                        futures.append(('Elasticsearch', executor.submit(self.load_batch_to_elasticsearch, cleaned_batch)))
                    
                    if 'PostgreSQL' in self.databases:
                        futures.append(('PostgreSQL', executor.submit(self.load_batch_to_postgresql, cleaned_batch)))
                    
                    if 'DuckDB' in self.databases:
                        futures.append(('DuckDB', executor.submit(self.load_batch_to_duckdb, cleaned_batch)))
                    
                    # Collect results
                    for db_name, future in futures:
                        try:
                            loaded_count = future.result(timeout=300)  # 5 minute timeout
                            self.loaded_records[db_name] += loaded_count
                            print(f"   ✅ {db_name}: {loaded_count} records loaded")
                        except Exception as e:
                            print(f"   ❌ {db_name}: {e}")
                
                processed_batches += 1
                
                # Progress report
                if processed_batches % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = processed_batches / total_batches * 100
                    print(f"\n📊 Progress: {progress:.1f}% ({processed_batches}/{total_batches} batches)")
                    print(f"⏱️ Elapsed: {elapsed/60:.1f} minutes")
                    print("📈 Records loaded by database:")
                    for db, count in self.loaded_records.items():
                        print(f"   {db}: {count:,}")
                    print()
                
            except Exception as e:
                print(f"❌ Batch loading failed at offset {offset}: {e}")
                continue
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 DATA LOADING COMPLETED")
        print("="*80)
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Processed batches: {processed_batches}")
        print("\n📈 Final record counts:")
        for db_name, count in self.loaded_records.items():
            if db_name in self.databases:
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
    loader = Chicago311DataLoader()
    
    try:
        loader.setup_database_connections()
        
        if not loader.databases:
            print("❌ No databases connected. Exiting.")
            return
        
        # Load the data
        loader.load_data_parallel()
        
        # Verify the loading
        verification_results = loader.verify_data_loading()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'loaded_records': loader.loaded_records,
            'verification_results': verification_results,
            'total_target_records': loader.total_records
        }
        
        with open('data_loading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Data loading results saved to 'data_loading_results.json'")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
    finally:
        loader.close_connections()

if __name__ == "__main__":
    main()