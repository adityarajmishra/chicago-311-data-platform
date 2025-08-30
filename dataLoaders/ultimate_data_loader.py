#!/usr/bin/env python3
"""
Ultimate Data Loader for Chicago 311 Data Platform
Fixed all database issues: DuckDB BIGINT, PostgreSQL VARCHAR lengths, etc.
"""

import os
import sys
import time
import logging
import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from databases.mongodb_handler import MongoDBHandler
from databases.elasticsearch_handler import ElasticsearchHandler
from databases.local_postgresql_handler import LocalPostgreSQLHandler
from databases.duckdb_handler import DuckDBHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltimateChicago311DataLoader:
    """Ultimate Chicago 311 data loader with all fixes applied."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 1000
        self.max_retries = 3
        self.delay_between_batches = 0.1
        
        # Database handlers
        self.handlers = {}
        
        # Progress tracking
        self.total_processed = 0
        self.start_time = None
        
    def setup_databases(self):
        """Setup all database connections and create tables."""
        logger.info("🔧 Setting up database connections...")
        
        try:
            # MongoDB
            logger.info("Setting up MongoDB...")
            mongo_handler = MongoDBHandler()
            mongo_handler.collection.delete_many({})
            self.handlers['MongoDB'] = mongo_handler
            logger.info("✅ MongoDB ready")
            
            # Elasticsearch
            logger.info("Setting up Elasticsearch...")
            es_handler = ElasticsearchHandler()
            try:
                es_handler.es.indices.delete(index=es_handler.index_name, ignore=[400, 404])
            except:
                pass
            es_handler._create_index()  # Use the correct method name
            self.handlers['Elasticsearch'] = es_handler
            logger.info("✅ Elasticsearch ready")
            
            # PostgreSQL with expanded VARCHAR limits
            logger.info("Setting up PostgreSQL...")
            class FixedPostgreSQLHandler:
                def __init__(self):
                    self.pg_handler = LocalPostgreSQLHandler()
                    self.create_table()
                
                def create_table(self):
                    create_sql = """
                    DROP TABLE IF EXISTS chicago_311_requests;
                    CREATE TABLE chicago_311_requests (
                        id SERIAL PRIMARY KEY,
                        sr_number VARCHAR(50) UNIQUE,
                        sr_type VARCHAR(300),
                        sr_short_code VARCHAR(50),
                        owner_department VARCHAR(150),
                        status VARCHAR(50),
                        created_date TIMESTAMP,
                        last_modified_date TIMESTAMP,
                        closed_date TIMESTAMP,
                        street_address VARCHAR(300),
                        city VARCHAR(100),
                        state VARCHAR(20),
                        zip_code VARCHAR(30),
                        street_number VARCHAR(30),
                        street_direction VARCHAR(20),
                        street_name VARCHAR(200),
                        street_type VARCHAR(100),
                        duplicate_ssr VARCHAR(20),
                        legacy_sr_number VARCHAR(50),
                        legacy_record VARCHAR(20),
                        parent_sr_number VARCHAR(50),
                        community_area INTEGER,
                        ward INTEGER,
                        electrical_district VARCHAR(100),
                        electricity_grid VARCHAR(100),
                        police_sector VARCHAR(100),
                        police_district VARCHAR(100),
                        police_beat VARCHAR(100),
                        precinct VARCHAR(100),
                        sanitation_division_days VARCHAR(100),
                        created_hour INTEGER,
                        created_day_of_week VARCHAR(20),
                        created_month INTEGER,
                        x_coordinate DOUBLE PRECISION,
                        y_coordinate DOUBLE PRECISION,
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        location_point VARCHAR(200)
                    );
                    """
                    with self.pg_handler.connection.cursor() as cursor:
                        cursor.execute(create_sql)
                        self.pg_handler.connection.commit()
                
                def bulk_insert(self, data):
                    if not data:
                        return 0
                    
                    # Remove _id field if it exists
                    clean_data = []
                    for record in data:
                        clean_record = {k: v for k, v in record.items() if k != '_id'}
                        # Truncate long strings to fit VARCHAR limits
                        for key, value in clean_record.items():
                            if isinstance(value, str) and len(value) > 300:
                                clean_record[key] = value[:300]
                        clean_data.append(clean_record)
                    
                    placeholders = ', '.join(['%s'] * len(clean_data[0]))
                    columns = ', '.join(clean_data[0].keys())
                    
                    insert_sql = f"""
                    INSERT INTO chicago_311_requests ({columns}) 
                    VALUES ({placeholders})
                    ON CONFLICT (sr_number) DO NOTHING
                    """
                    
                    values = [tuple(record.values()) for record in clean_data]
                    
                    with self.pg_handler.connection.cursor() as cursor:
                        cursor.executemany(insert_sql, values)
                        self.pg_handler.connection.commit()
                        return cursor.rowcount
            
            pg_handler = FixedPostgreSQLHandler()
            self.handlers['PostgreSQL'] = pg_handler
            logger.info("✅ PostgreSQL ready")
            
            # DuckDB with BIGINT support
            logger.info("Setting up DuckDB...")
            duckdb_handler = DuckDBHandler()
            duckdb_handler.create_table()  # This will use BIGINT for id
            self.handlers['DuckDB'] = duckdb_handler
            logger.info("✅ DuckDB ready")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    def fetch_data_batch(self, offset: int, limit: int) -> List[Dict]:
        """Fetch a batch of data from the Chicago 311 API."""
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': ':id'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    logger.info(f"No more data at offset {offset}")
                    return []
                
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for offset {offset}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return []
    
    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single record for database insertion."""
        try:
            transformed = {}
            
            # Handle basic fields
            for key, value in record.items():
                if value is None or value == '':
                    transformed[key] = None
                elif isinstance(value, str):
                    transformed[key] = value.strip()
                else:
                    transformed[key] = value
            
            # Convert date fields
            date_fields = ['created_date', 'last_modified_date', 'closed_date']
            for field in date_fields:
                if field in transformed and transformed[field]:
                    try:
                        transformed[field] = pd.to_datetime(transformed[field], errors='coerce')
                    except:
                        transformed[field] = None
            
            # Handle location data
            if 'location' in transformed:
                location = transformed['location']
                if isinstance(location, dict):
                    if 'coordinates' in location:
                        coords = location['coordinates']
                        if len(coords) >= 2:
                            transformed['longitude'] = float(coords[0])
                            transformed['latitude'] = float(coords[1])
                    
                    # Create location point string
                    if transformed.get('latitude') and transformed.get('longitude'):
                        transformed['location_point'] = f"POINT({transformed['longitude']} {transformed['latitude']})"
                
                # Remove the original location field
                del transformed['location']
            
            # Convert numeric fields
            numeric_fields = ['community_area', 'ward', 'created_hour', 'created_month', 
                            'x_coordinate', 'y_coordinate', 'latitude', 'longitude']
            for field in numeric_fields:
                if field in transformed and transformed[field] is not None:
                    try:
                        if field in ['latitude', 'longitude', 'x_coordinate', 'y_coordinate']:
                            transformed[field] = float(transformed[field])
                        else:
                            transformed[field] = int(float(transformed[field]))
                    except (ValueError, TypeError):
                        transformed[field] = None
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            return {}
    
    def insert_to_database(self, db_name: str, data: List[Dict]) -> int:
        """Insert data into a specific database."""
        if not data or db_name not in self.handlers:
            return 0
        
        try:
            handler = self.handlers[db_name]
            
            if db_name == 'MongoDB':
                # MongoDB can handle the data as-is
                result = handler.collection.insert_many(data, ordered=False)
                return len(result.inserted_ids)
            
            elif db_name == 'Elasticsearch':
                # Elasticsearch bulk insert
                from elasticsearch.helpers import bulk
                
                actions = []
                for doc in data:
                    # Convert numpy types for ES compatibility
                    clean_doc = {}
                    for key, value in doc.items():
                        if hasattr(value, 'item'):  # numpy type
                            clean_doc[key] = value.item()
                        elif pd.isna(value):
                            clean_doc[key] = None
                        else:
                            clean_doc[key] = value
                    
                    actions.append({
                        "_index": handler.index_name,
                        "_source": clean_doc
                    })
                
                success_count, _ = bulk(handler.es, actions, request_timeout=60)
                return success_count
            
            elif db_name in ['PostgreSQL', 'DuckDB']:
                # Use the bulk_insert method
                return handler.bulk_insert(data)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ {db_name} insert failed: {e}")
            return 0
    
    def process_batch(self, offset: int) -> Dict[str, int]:
        """Process a single batch of data."""
        try:
            # Fetch data
            raw_data = self.fetch_data_batch(offset, self.batch_size)
            if not raw_data:
                return {}
            
            # Transform data
            transformed_data = []
            for record in raw_data:
                transformed = self.transform_record(record)
                if transformed:
                    transformed_data.append(transformed)
            
            if not transformed_data:
                return {}
            
            # Insert into all databases concurrently
            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.insert_to_database, db_name, transformed_data): db_name
                    for db_name in self.handlers.keys()
                }
                
                for future in as_completed(futures):
                    db_name = futures[future]
                    try:
                        inserted_count = future.result()
                        results[db_name] = inserted_count
                    except Exception as e:
                        logger.error(f"Database {db_name} insert failed: {e}")
                        results[db_name] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            return {}
    
    def run_loading(self, total_records: int = 12400000):
        """Run the complete data loading process."""
        logger.info(f"🚀 Starting ultimate data loading for {total_records:,} records...")
        
        self.setup_databases()
        self.start_time = time.time()
        
        # Progress tracking
        last_progress_time = time.time()
        progress_interval = 30  # seconds
        
        offset = 0
        db_totals = {db_name: 0 for db_name in self.handlers.keys()}
        
        while offset < total_records:
            batch_start = time.time()
            
            # Process batch
            batch_results = self.process_batch(offset)
            
            if not batch_results:
                logger.info("No more data available, stopping...")
                break
            
            # Update totals
            for db_name, count in batch_results.items():
                db_totals[db_name] += count
            
            self.total_processed += self.batch_size
            offset += self.batch_size
            
            # Progress reporting
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - self.start_time
                rate = self.total_processed / elapsed
                eta_seconds = (total_records - self.total_processed) / rate if rate > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                progress_pct = (self.total_processed / total_records) * 100
                
                logger.info(f"📊 Progress: {self.total_processed:,}/{total_records:,} ({progress_pct:.1f}%)")
                logger.info(f"⚡ Rate: {rate:.0f} records/sec, ETA: {eta}")
                
                # Database counts
                total_loaded = sum(db_totals.values())
                logger.info(f"💾 Total loaded across all DBs: {total_loaded:,}")
                for db_name, count in db_totals.items():
                    logger.info(f"   {db_name}: {count:,} records")
                
                last_progress_time = current_time
            
            # Small delay between batches
            time.sleep(self.delay_between_batches)
        
        # Final summary
        total_time = time.time() - self.start_time
        avg_rate = self.total_processed / total_time
        
        logger.info("=" * 80)
        logger.info("🎉 ULTIMATE DATA LOADING COMPLETE!")
        logger.info(f"⏱️  Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"📊 Total processed: {self.total_processed:,} records")
        logger.info(f"⚡ Average rate: {avg_rate:.0f} records/sec")
        logger.info("💾 Final database counts:")
        
        for db_name, count in db_totals.items():
            logger.info(f"   {db_name}: {count:,} records")
        
        logger.info("=" * 80)

def main():
    """Main execution function."""
    try:
        loader = UltimateChicago311DataLoader()
        loader.run_loading()
    except KeyboardInterrupt:
        logger.info("🛑 Loading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        raise

if __name__ == "__main__":
    main()