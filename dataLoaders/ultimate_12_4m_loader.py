#!/usr/bin/env python3
"""
Ultimate 12.4M Chicago 311 Data Loader
Fixes ALL database issues and loads complete dataset
"""

import os
import sys
import time
import logging
import requests
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import traceback

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from databases.mongodb_handler import MongoDBHandler
from databases.elasticsearch_handler import ElasticsearchHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Ultimate12_4MLoader:
    """Ultimate loader for complete 12.4M Chicago 311 dataset."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 5000  # Larger batches for efficiency
        self.max_retries = 3
        self.delay_between_batches = 0.5
        
        # Database handlers
        self.handlers = {}
        
        # Progress tracking
        self.total_processed = 0
        self.start_time = None
        
    def setup_databases(self):
        """Setup all database connections with proper fixes."""
        logger.info("🔧 Setting up all 4 database connections...")
        
        # 1. MongoDB Setup
        try:
            logger.info("Setting up MongoDB...")
            mongo_handler = MongoDBHandler()
            # Clear existing data
            mongo_handler.collection.delete_many({})
            self.handlers['MongoDB'] = mongo_handler
            logger.info("✅ MongoDB ready")
        except Exception as e:
            logger.error(f"❌ MongoDB setup failed: {e}")
            
        # 2. Elasticsearch Setup with NumPy 2.0 fix
        try:
            logger.info("Setting up Elasticsearch...")
            es_handler = ElasticsearchHandler()
            try:
                es_handler.es.indices.delete(index=es_handler.index_name, ignore=[400, 404])
            except:
                pass
            es_handler._create_index()
            self.handlers['Elasticsearch'] = es_handler
            logger.info("✅ Elasticsearch ready")
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup failed: {e}")
            
        # 3. PostgreSQL Setup (Local)
        try:
            logger.info("Setting up PostgreSQL...")
            import psycopg2
            
            class LocalPostgreSQLHandler:
                def __init__(self):
                    # Try to connect to local PostgreSQL
                    try:
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="postgres",
                            user="postgres",
                            password="postgres"
                        )
                        self.connection.autocommit = True
                        logger.info("Connected to local PostgreSQL")
                    except:
                        # Create database if needed
                        logger.info("Creating PostgreSQL setup...")
                        os.system("createdb chicago_311 2>/dev/null || true")
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="chicago_311",
                            user="postgres",
                            password=""
                        )
                        self.connection.autocommit = True
                    
                    self.create_table()
                
                def create_table(self):
                    create_sql = """
                    DROP TABLE IF EXISTS chicago_311_requests;
                    CREATE TABLE chicago_311_requests (
                        id SERIAL PRIMARY KEY,
                        sr_number VARCHAR(50) UNIQUE,
                        sr_type VARCHAR(500),
                        sr_short_code VARCHAR(50),
                        owner_department VARCHAR(200),
                        status VARCHAR(50),
                        created_date TIMESTAMP,
                        last_modified_date TIMESTAMP,
                        closed_date TIMESTAMP,
                        street_address VARCHAR(500),
                        city VARCHAR(100),
                        state VARCHAR(20),
                        zip_code VARCHAR(50),
                        street_number VARCHAR(50),
                        street_direction VARCHAR(50),
                        street_name VARCHAR(300),
                        street_type VARCHAR(100),
                        duplicate_ssr VARCHAR(50),
                        legacy_sr_number VARCHAR(50),
                        legacy_record VARCHAR(50),
                        parent_sr_number VARCHAR(50),
                        community_area INTEGER,
                        ward INTEGER,
                        electrical_district VARCHAR(200),
                        electricity_grid VARCHAR(200),
                        police_sector VARCHAR(200),
                        police_district VARCHAR(200),
                        police_beat VARCHAR(200),
                        precinct VARCHAR(200),
                        sanitation_division_days VARCHAR(200),
                        created_hour INTEGER,
                        created_day_of_week VARCHAR(50),
                        created_month INTEGER,
                        x_coordinate DOUBLE PRECISION,
                        y_coordinate DOUBLE PRECISION,
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        location_point TEXT,
                        origin VARCHAR(200),
                        type_of_service_request VARCHAR(500),
                        most_recent_action VARCHAR(500),
                        current_activity VARCHAR(500),
                        number_of_days_pending INTEGER,
                        completion_date TIMESTAMP,
                        due_date TIMESTAMP,
                        ssa VARCHAR(50),
                        channel VARCHAR(100),
                        historical_wards_03_15 INTEGER,
                        zip_codes INTEGER,
                        community_areas INTEGER,
                        census_tracts BIGINT,
                        wards INTEGER
                    );
                    CREATE INDEX IF NOT EXISTS idx_sr_number ON chicago_311_requests(sr_number);
                    CREATE INDEX IF NOT EXISTS idx_created_date ON chicago_311_requests(created_date);
                    CREATE INDEX IF NOT EXISTS idx_status ON chicago_311_requests(status);
                    """
                    
                    with self.connection.cursor() as cursor:
                        cursor.execute(create_sql)
                
                def bulk_insert(self, data):
                    if not data:
                        return 0
                    
                    # Clean data for PostgreSQL
                    clean_data = []
                    for record in data:
                        clean_record = {}
                        for key, value in record.items():
                            if key == '_id':  # Skip MongoDB ObjectId
                                continue
                            if isinstance(value, str) and len(value) > 500:
                                clean_record[key] = value[:500]
                            elif pd.isna(value) or value == '':
                                clean_record[key] = None
                            else:
                                clean_record[key] = value
                        clean_data.append(clean_record)
                    
                    if not clean_data:
                        return 0
                    
                    # Dynamic insert based on available columns
                    columns = list(clean_data[0].keys())
                    placeholders = ', '.join(['%s'] * len(columns))
                    columns_str = ', '.join(columns)
                    
                    insert_sql = f"""
                    INSERT INTO chicago_311_requests ({columns_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (sr_number) DO NOTHING
                    """
                    
                    values = [tuple(record[col] for col in columns) for record in clean_data]
                    
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany(insert_sql, values)
                            return cursor.rowcount
                    except Exception as e:
                        logger.error(f"PostgreSQL insert error: {e}")
                        return 0
            
            pg_handler = LocalPostgreSQLHandler()
            self.handlers['PostgreSQL'] = pg_handler
            logger.info("✅ PostgreSQL ready")
        except Exception as e:
            logger.error(f"❌ PostgreSQL setup failed: {e}")
            
        # 4. DuckDB Setup
        try:
            logger.info("Setting up DuckDB...")
            import duckdb
            
            class DuckDBHandler:
                def __init__(self):
                    self.db_path = "chicago_311.duckdb"
                    self.conn = duckdb.connect(self.db_path)
                    self.create_table()
                
                def create_table(self):
                    create_sql = """
                    DROP TABLE IF EXISTS chicago_311_requests;
                    CREATE TABLE chicago_311_requests (
                        id INTEGER,
                        sr_number VARCHAR,
                        sr_type VARCHAR,
                        sr_short_code VARCHAR,
                        owner_department VARCHAR,
                        status VARCHAR,
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
                        duplicate_ssr VARCHAR,
                        legacy_sr_number VARCHAR,
                        legacy_record VARCHAR,
                        parent_sr_number VARCHAR,
                        community_area INTEGER,
                        ward INTEGER,
                        electrical_district VARCHAR,
                        electricity_grid VARCHAR,
                        police_sector VARCHAR,
                        police_district VARCHAR,
                        police_beat VARCHAR,
                        precinct VARCHAR,
                        sanitation_division_days VARCHAR,
                        created_hour INTEGER,
                        created_day_of_week VARCHAR,
                        created_month INTEGER,
                        x_coordinate DOUBLE,
                        y_coordinate DOUBLE,
                        latitude DOUBLE,
                        longitude DOUBLE,
                        location_point VARCHAR,
                        origin VARCHAR,
                        type_of_service_request VARCHAR,
                        most_recent_action VARCHAR,
                        current_activity VARCHAR,
                        number_of_days_pending INTEGER,
                        completion_date TIMESTAMP,
                        due_date TIMESTAMP,
                        ssa VARCHAR,
                        channel VARCHAR,
                        historical_wards_03_15 INTEGER,
                        zip_codes INTEGER,
                        community_areas INTEGER,
                        census_tracts BIGINT,
                        wards INTEGER
                    );
                    """
                    self.conn.execute(create_sql)
                
                def bulk_insert(self, data):
                    if not data:
                        return 0
                    
                    # Convert to DataFrame for DuckDB
                    df = pd.DataFrame(data)
                    df = df.drop(columns=['_id'], errors='ignore')  # Remove MongoDB ObjectId
                    
                    # Handle data types
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).replace({'nan': None, 'None': None})
                    
                    try:
                        # Use INSERT OR IGNORE for DuckDB
                        self.conn.register('temp_data', df)
                        self.conn.execute("""
                            INSERT OR IGNORE INTO chicago_311_requests 
                            SELECT * FROM temp_data
                        """)
                        return len(df)
                    except Exception as e:
                        logger.error(f"DuckDB insert error: {e}")
                        return 0
            
            duckdb_handler = DuckDBHandler()
            self.handlers['DuckDB'] = duckdb_handler
            logger.info("✅ DuckDB ready")
        except Exception as e:
            logger.error(f"❌ DuckDB setup failed: {e}")
        
        logger.info(f"🎯 Successfully set up {len(self.handlers)} databases: {list(self.handlers.keys())}")
    
    def fetch_data_batch(self, offset: int, limit: int) -> List[Dict]:
        """Fetch a batch from Chicago 311 API."""
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': ':id'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    logger.info(f"No more data at offset {offset}")
                    return []
                
                logger.info(f"✅ Fetched batch: offset={offset}, count={len(data)}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for offset {offset}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return []
    
    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API record for database insertion."""
        try:
            transformed = {}
            
            # Handle all API fields
            field_mapping = {
                'sr_number': record.get('sr_number'),
                'sr_type': record.get('sr_type'),
                'sr_short_code': record.get('sr_short_code'),
                'owner_department': record.get('owner_department'),
                'status': record.get('status'),
                'origin': record.get('origin'),
                'type_of_service_request': record.get('type_of_service_request'),
                'most_recent_action': record.get('most_recent_action'),
                'current_activity': record.get('current_activity'),
                'ssa': record.get('ssa'),
                'channel': record.get('channel'),
                'street_address': record.get('street_address'),
                'city': record.get('city'),
                'state': record.get('state'),
                'zip_code': record.get('zip_code'),
                'street_number': record.get('street_number'),
                'street_direction': record.get('street_direction'),
                'street_name': record.get('street_name'),
                'street_type': record.get('street_type'),
                'duplicate_ssr': record.get('duplicate_ssr'),
                'legacy_sr_number': record.get('legacy_sr_number'),
                'legacy_record': record.get('legacy_record'),
                'parent_sr_number': record.get('parent_sr_number'),
                'electrical_district': record.get('electrical_district'),
                'electricity_grid': record.get('electricity_grid'),
                'police_sector': record.get('police_sector'),
                'police_district': record.get('police_district'),
                'police_beat': record.get('police_beat'),
                'precinct': record.get('precinct'),
                'sanitation_division_days': record.get('sanitation_division_days'),
                'created_day_of_week': record.get('created_day_of_week'),
                'location_point': record.get('location_point')
            }
            
            # Add computed region fields
            field_mapping.update({
                'historical_wards_03_15': record.get(':@computed_region_rpca_8um6'),
                'zip_codes': record.get(':@computed_region_6mkv_f3dw'),
                'community_areas': record.get(':@computed_region_vrxf_vc4k'),
                'census_tracts': record.get(':@computed_region_bdys_3d7i'),
                'wards': record.get(':@computed_region_43wa_7qmu')
            })
            
            for key, value in field_mapping.items():
                if value is None or value == '':
                    transformed[key] = None
                elif isinstance(value, str):
                    transformed[key] = value.strip()
                else:
                    transformed[key] = value
            
            # Handle date fields
            date_fields = ['created_date', 'last_modified_date', 'closed_date', 'completion_date', 'due_date']
            for field in date_fields:
                if field in record and record[field]:
                    try:
                        transformed[field] = pd.to_datetime(record[field], errors='coerce')
                    except:
                        transformed[field] = None
            
            # Handle numeric fields
            numeric_fields = ['community_area', 'ward', 'created_hour', 'created_month', 
                            'x_coordinate', 'y_coordinate', 'latitude', 'longitude',
                            'number_of_days_pending', 'historical_wards_03_15', 
                            'zip_codes', 'community_areas', 'census_tracts', 'wards']
            
            for field in numeric_fields:
                if field in record and record[field] is not None:
                    try:
                        if field in ['latitude', 'longitude', 'x_coordinate', 'y_coordinate']:
                            transformed[field] = float(record[field])
                        elif field == 'census_tracts':
                            transformed[field] = int(float(record[field])) if record[field] else None
                        else:
                            transformed[field] = int(float(record[field])) if record[field] else None
                    except (ValueError, TypeError):
                        transformed[field] = None
            
            # Handle location data
            if 'location' in record and record['location']:
                location = record['location']
                if isinstance(location, dict) and 'coordinates' in location:
                    coords = location['coordinates']
                    if len(coords) >= 2:
                        transformed['longitude'] = float(coords[0])
                        transformed['latitude'] = float(coords[1])
                        transformed['location_point'] = f"POINT({coords[0]} {coords[1]})"
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            return {}
    
    def insert_to_database(self, db_name: str, data: List[Dict]) -> int:
        """Insert data into specific database."""
        if not data or db_name not in self.handlers:
            return 0
        
        try:
            handler = self.handlers[db_name]
            
            if db_name == 'MongoDB':
                try:
                    result = handler.collection.insert_many(data, ordered=False)
                    return len(result.inserted_ids)
                except Exception as e:
                    logger.error(f"MongoDB insert error: {e}")
                    return 0
            
            elif db_name == 'Elasticsearch':
                try:
                    from elasticsearch.helpers import bulk
                    
                    actions = []
                    for doc in data:
                        clean_doc = {}
                        for key, value in doc.items():
                            # Handle NumPy 2.0 compatibility
                            if hasattr(value, 'item'):  # numpy type
                                clean_doc[key] = value.item()
                            elif pd.isna(value):
                                clean_doc[key] = None
                            elif isinstance(value, (np.floating, float)):
                                clean_doc[key] = float(value)
                            elif isinstance(value, (np.integer, int)):
                                clean_doc[key] = int(value)
                            else:
                                clean_doc[key] = value
                        
                        actions.append({
                            "_index": handler.index_name,
                            "_source": clean_doc
                        })
                    
                    success_count, _ = bulk(handler.es, actions, request_timeout=120)
                    return success_count
                except Exception as e:
                    logger.error(f"Elasticsearch insert error: {e}")
                    return 0
            
            else:  # PostgreSQL or DuckDB
                return handler.bulk_insert(data)
            
        except Exception as e:
            logger.error(f"❌ {db_name} insert failed: {e}")
            return 0
    
    def process_batch(self, offset: int) -> Dict[str, int]:
        """Process a single batch across all databases."""
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
                        logger.info(f"✅ {db_name}: {inserted_count} records inserted")
                    except Exception as e:
                        logger.error(f"❌ {db_name} insert failed: {e}")
                        results[db_name] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            return {}
    
    def run_complete_loading(self):
        """Load complete 12.4M dataset."""
        target_records = 12400000
        logger.info(f"🚀 Starting COMPLETE 12.4M record loading...")
        logger.info(f"📊 Target: {target_records:,} records")
        
        # Setup databases
        self.setup_databases()
        if not self.handlers:
            logger.error("❌ No databases available!")
            return
        
        self.start_time = time.time()
        
        # Progress tracking
        last_progress_time = time.time()
        progress_interval = 60  # Report every minute
        
        offset = 0
        db_totals = {db_name: 0 for db_name in self.handlers.keys()}
        
        try:
            while offset < target_records:
                batch_start = time.time()
                
                # Process batch
                batch_results = self.process_batch(offset)
                
                if not batch_results:
                    logger.warning("❌ No data returned, stopping...")
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
                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (target_records - self.total_processed) / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(eta_seconds))
                    
                    progress_pct = (self.total_processed / target_records) * 100
                    
                    logger.info("=" * 80)
                    logger.info(f"📊 PROGRESS: {self.total_processed:,}/{target_records:,} ({progress_pct:.1f}%)")
                    logger.info(f"⚡ Rate: {rate:.0f} records/sec | ETA: {eta}")
                    logger.info(f"💾 Database totals:")
                    for db_name, count in db_totals.items():
                        logger.info(f"   {db_name}: {count:,} records")
                    logger.info("=" * 80)
                    
                    last_progress_time = current_time
                
                # Adaptive delay
                time.sleep(self.delay_between_batches)
                
        except KeyboardInterrupt:
            logger.info("🛑 Loading interrupted by user")
        except Exception as e:
            logger.error(f"❌ Loading failed: {e}")
            traceback.print_exc()
        
        # Final summary
        total_time = time.time() - self.start_time
        avg_rate = self.total_processed / total_time if total_time > 0 else 0
        
        logger.info("=" * 80)
        logger.info("🎉 COMPLETE DATA LOADING FINISHED!")
        logger.info(f"⏱️  Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"📊 Total processed: {self.total_processed:,} records")
        logger.info(f"⚡ Average rate: {avg_rate:.0f} records/sec")
        logger.info("💾 Final database counts:")
        
        total_loaded = 0
        for db_name, count in db_totals.items():
            logger.info(f"   {db_name}: {count:,} records")
            total_loaded += count
        
        logger.info(f"🎯 Total loaded across all databases: {total_loaded:,}")
        logger.info("=" * 80)
        
        return db_totals

def main():
    """Main execution."""
    try:
        loader = Ultimate12_4MLoader()
        results = loader.run_complete_loading()
        
        logger.info("🚀 Ready for comprehensive benchmarking and stress testing!")
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Loading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()