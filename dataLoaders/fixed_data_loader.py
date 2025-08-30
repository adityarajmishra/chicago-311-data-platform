#!/usr/bin/env python3
"""
Fixed Data Loader for All 4 Databases
Fixes all data type issues and loads complete 12.4M Chicago 311 records
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import logging
import requests
import pandas as pd
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

# Database handlers
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedDataLoader:
    """Fixed data loader that handles all data type issues."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 5000
        self.max_retries = 3
        self.total_records = 0
        self.loaded_records = {}
        self.databases = {}
        
    def setup_databases(self):
        """Setup all 4 databases with proper error handling."""
        print("🔄 Setting up all 4 databases...")
        
        # 1. MongoDB - Clear and setup
        try:
            mongo_handler = MongoDBHandler()
            mongo_handler.collection.delete_many({})
            self.databases['MongoDB'] = mongo_handler
            self.loaded_records['MongoDB'] = 0
            print("✅ MongoDB: Connected and cleared")
        except Exception as e:
            print(f"❌ MongoDB failed: {e}")
        
        # 2. Elasticsearch - Clear and setup
        try:
            es_handler = ElasticsearchHandler()
            try:
                es_handler.es.delete_by_query(
                    index=es_handler.index_name,
                    body={"query": {"match_all": {}}}
                )
            except:
                pass
            self.databases['Elasticsearch'] = es_handler
            self.loaded_records['Elasticsearch'] = 0
            print("✅ Elasticsearch: Connected and cleared")
        except Exception as e:
            print(f"❌ Elasticsearch failed: {e}")
        
        # 3. PostgreSQL - Fixed version
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            class FixedPostgreSQLHandler:
                def __init__(self):
                    self.setup_database()
                
                def setup_database(self):
                    # Connect to postgres to create/recreate database
                    conn = psycopg2.connect(
                        host="localhost",
                        port=5432,
                        database="postgres",
                        user="postgres",
                        password=""
                    )
                    conn.autocommit = True
                    cursor = conn.cursor()
                    
                    cursor.execute("DROP DATABASE IF EXISTS chicago_311_complete")
                    cursor.execute("CREATE DATABASE chicago_311_complete")
                    cursor.close()
                    conn.close()
                    
                    # Connect to new database
                    self.connection = psycopg2.connect(
                        host="localhost",
                        port=5432,
                        database="chicago_311_complete",
                        user="postgres",
                        password="",
                        cursor_factory=RealDictCursor
                    )
                    
                    self.create_table()
                
                def create_table(self):
                    create_sql = """
                    CREATE TABLE chicago_311_requests (
                        id SERIAL PRIMARY KEY,
                        sr_number VARCHAR(50) UNIQUE,
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
                    
                    CREATE INDEX idx_sr_number ON chicago_311_requests(sr_number);
                    CREATE INDEX idx_sr_type ON chicago_311_requests(sr_type);
                    CREATE INDEX idx_status ON chicago_311_requests(status);
                    CREATE INDEX idx_created_date ON chicago_311_requests(created_date);
                    CREATE INDEX idx_ward ON chicago_311_requests(ward);
                    """
                    
                    cursor = self.connection.cursor()
                    cursor.execute(create_sql)
                    self.connection.commit()
                    cursor.close()
                
                def insert_batch(self, batch_data):
                    cursor = self.connection.cursor()
                    try:
                        insert_sql = """
                        INSERT INTO chicago_311_requests (
                            sr_number, sr_type, sr_short_code, owner_department, status, origin,
                            created_date, last_modified_date, closed_date, street_address, city, state,
                            zip_code, street_number, street_direction, street_name, street_type,
                            duplicate, legacy_record, latitude, longitude, ward, community_area, police_district
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (sr_number) DO NOTHING
                        """
                        
                        values = []
                        for record in batch_data:
                            values.append((
                                record.get('sr_number'),
                                record.get('sr_type'),
                                record.get('sr_short_code'),
                                record.get('owner_department'),
                                record.get('status'),
                                record.get('origin'),
                                record.get('created_date'),
                                record.get('last_modified_date'),
                                record.get('closed_date'),
                                record.get('street_address'),
                                record.get('city'),
                                record.get('state'),
                                record.get('zip_code'),
                                record.get('street_number'),
                                record.get('street_direction'),
                                record.get('street_name'),
                                record.get('street_type'),
                                record.get('duplicate'),
                                record.get('legacy_record'),
                                record.get('latitude'),
                                record.get('longitude'),
                                record.get('ward'),
                                record.get('community_area'),
                                record.get('police_district')
                            ))
                        
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
            
            pg_handler = FixedPostgreSQLHandler()
            self.databases['PostgreSQL'] = pg_handler
            self.loaded_records['PostgreSQL'] = 0
            print("✅ PostgreSQL: Connected with fixed schema")
            
        except Exception as e:
            print(f"❌ PostgreSQL failed: {e}")
            print("   Install: brew install postgresql && brew services start postgresql")
            print("   Then run: createuser -s postgres")
        
        # 4. DuckDB - Fixed version
        try:
            import duckdb
            
            class FixedDuckDBHandler:
                def __init__(self):
                    self.db_path = "chicago_311_complete.duckdb"
                    if os.path.exists(self.db_path):
                        os.remove(self.db_path)
                    
                    self.connection = duckdb.connect(self.db_path)
                    self.create_table()
                
                def create_table(self):
                    create_sql = """
                    CREATE TABLE chicago_311_requests (
                        id INTEGER PRIMARY KEY,
                        sr_number VARCHAR UNIQUE,
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
                        # Prepare data with explicit ID
                        processed_data = []
                        for i, record in enumerate(batch_data):
                            row = {
                                'id': int(time.time() * 1000000) + i,  # Unique ID
                                'sr_number': record.get('sr_number'),
                                'sr_type': record.get('sr_type'),
                                'sr_short_code': record.get('sr_short_code'),
                                'owner_department': record.get('owner_department'),
                                'status': record.get('status'),
                                'origin': record.get('origin'),
                                'created_date': record.get('created_date'),
                                'last_modified_date': record.get('last_modified_date'),
                                'closed_date': record.get('closed_date'),
                                'street_address': record.get('street_address'),
                                'city': record.get('city'),
                                'state': record.get('state'),
                                'zip_code': record.get('zip_code'),
                                'street_number': record.get('street_number'),
                                'street_direction': record.get('street_direction'),
                                'street_name': record.get('street_name'),
                                'street_type': record.get('street_type'),
                                'duplicate': record.get('duplicate'),
                                'legacy_record': record.get('legacy_record'),
                                'latitude': record.get('latitude'),
                                'longitude': record.get('longitude'),
                                'ward': record.get('ward'),
                                'community_area': record.get('community_area'),
                                'police_district': record.get('police_district')
                            }
                            processed_data.append(row)
                        
                        df = pd.DataFrame(processed_data)
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
                    try:
                        result = self.connection.execute("SELECT COUNT(*) FROM chicago_311_requests").fetchone()
                        return result[0] if result else 0
                    except:
                        return 0
                
                def close(self):
                    if self.connection:
                        self.connection.close()
            
            duckdb_handler = FixedDuckDBHandler()
            self.databases['DuckDB'] = duckdb_handler
            self.loaded_records['DuckDB'] = 0
            print("✅ DuckDB: Connected with fixed schema")
            
        except Exception as e:
            print(f"❌ DuckDB failed: {e}")
        
        if not self.databases:
            raise Exception("No databases available!")
        
        print(f"📊 Ready databases: {list(self.databases.keys())}")
        return len(self.databases)
    
    def fetch_data_batch(self, offset: int, limit: int) -> List[Dict]:
        """Fetch data from Chicago 311 API."""
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': 'sr_number'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return []
    
    def clean_record(self, record: Dict) -> Dict:
        """Clean record without ObjectId issues."""
        cleaned = {}
        
        # String fields
        string_fields = [
            'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 
            'status', 'origin', 'street_address', 'city', 'state', 'zip_code',
            'street_number', 'street_direction', 'street_name', 'street_type',
            'police_district'
        ]
        
        for field in string_fields:
            value = record.get(field)
            cleaned[field] = str(value).strip() if value else None
        
        # Integer fields
        for field in ['ward', 'community_area']:
            value = record.get(field)
            try:
                cleaned[field] = int(float(value)) if value else None
            except (ValueError, TypeError):
                cleaned[field] = None
        
        # Float fields
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
        
        return cleaned
    
    def load_batch_to_database(self, db_name: str, batch_data: List[Dict]) -> int:
        """Load batch to specific database."""
        if db_name not in self.databases:
            return 0
        
        try:
            handler = self.databases[db_name]
            
            if db_name == 'MongoDB':
                # Remove any _id fields to avoid ObjectId issues
                clean_batch = []
                for doc in batch_data:
                    clean_doc = {k: v for k, v in doc.items() if k != '_id'}
                    clean_batch.append(clean_doc)
                
                result = handler.collection.insert_many(clean_batch, ordered=False)
                return len(result.inserted_ids)
            
            elif db_name == 'Elasticsearch':
                from elasticsearch.helpers import bulk
                actions = []
                for doc in batch_data:
                    # Convert dates to strings for ES
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
                logger.error(f"{db_name} batch failed: {e}")
            return 0
        
        return 0
    
    def load_complete_dataset(self):
        """Load the complete 12.4M dataset."""
        print("🚀 Loading COMPLETE 12.4M Chicago 311 Dataset")
        print("=" * 60)
        
        # Get total record count
        try:
            response = requests.get(f"{self.api_url}?$select=count(*)", timeout=30)
            if response.status_code == 200:
                total_records = int(response.json()[0]['count'])
            else:
                total_records = 12_400_000
        except:
            total_records = 12_400_000
        
        print(f"📊 Target: {total_records:,} records")
        self.total_records = total_records
        
        # Load data
        processed_records = 0
        offset = 0
        start_time = time.time()
        last_progress_time = start_time
        consecutive_failures = 0
        max_failures = 10
        
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
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    for db_name in self.databases.keys():
                        futures[db_name] = executor.submit(
                            self.load_batch_to_database, db_name, cleaned_batch
                        )
                    
                    # Collect results
                    batch_results = {}
                    for db_name, future in futures.items():
                        try:
                            loaded_count = future.result(timeout=180)
                            batch_results[db_name] = loaded_count
                            self.loaded_records[db_name] += loaded_count
                        except Exception as e:
                            batch_results[db_name] = 0
                            logger.error(f"{db_name}: {e}")
                
                # Show results
                for db_name, count in batch_results.items():
                    print(f"   ✅ {db_name}: +{count}")
                
                processed_records += len(batch_data)
                offset += len(batch_data)
                consecutive_failures = 0
                
                # Progress report every 100K records
                if processed_records % 100000 == 0 or time.time() - last_progress_time > 300:
                    elapsed = time.time() - start_time
                    progress = (processed_records / total_records) * 100
                    rate = processed_records / elapsed if elapsed > 0 else 0
                    eta_hours = (total_records - processed_records) / rate / 3600 if rate > 0 else 0
                    
                    print(f"\n📊 Progress: {progress:.1f}% ({processed_records:,}/{total_records:,})")
                    print(f"⏱️ Rate: {rate:.0f} records/sec, ETA: {eta_hours:.1f} hours")
                    print("📈 Loaded by database:")
                    for db, count in self.loaded_records.items():
                        print(f"   {db}: {count:,}")
                    print()
                    
                    last_progress_time = time.time()
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"❌ Batch failed: {e}")
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    print(f"⏸️ Retrying... ({consecutive_failures}/{max_failures})")
                    time.sleep(60)
                else:
                    print("💥 Too many failures!")
                    break
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("🎉 DATA LOADING COMPLETED!")
        print(f"{'='*60}")
        print(f"⏱️ Total time: {total_time/3600:.1f} hours")
        print(f"📊 Records processed: {processed_records:,}")
        print(f"📈 Average rate: {processed_records/total_time:.0f} records/second")
        print("\n📈 Final counts by database:")
        for db_name, count in self.loaded_records.items():
            print(f"   {db_name}: {count:,} records")
    
    def verify_all_databases(self):
        """Verify data in all databases."""
        print(f"\n🔍 Verifying all databases...")
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
                verification_results[db_name] = -1
                print(f"❌ {db_name}: Verification failed - {e}")
        
        return verification_results
    
    def close_connections(self):
        """Close all connections."""
        for handler in self.databases.values():
            try:
                handler.close()
            except:
                pass

def main():
    """Load complete dataset into all 4 databases."""
    loader = FixedDataLoader()
    
    try:
        # Setup databases
        db_count = loader.setup_databases()
        print(f"\n✅ Successfully connected to {db_count} databases")
        
        if db_count == 0:
            print("❌ No databases available. Please fix database setup.")
            return
        
        # Confirm before starting massive load
        print(f"\n⚠️  WARNING: This will load 12.4 MILLION records!")
        print("   This process will take several hours.")
        confirm = input("   Continue? (yes/no): ").lower().strip()
        
        if confirm != 'yes':
            print("Operation cancelled.")
            return
        
        # Load complete dataset
        loader.load_complete_dataset()
        
        # Verify results
        verification = loader.verify_all_databases()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'target_records': loader.total_records,
            'loaded_records': loader.loaded_records,
            'verification_results': verification,
            'databases_used': list(loader.databases.keys())
        }
        
        with open('complete_12_4m_loading_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Complete results saved to 'complete_12_4m_loading_results.json'")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close_connections()

if __name__ == "__main__":
    main()