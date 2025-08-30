#!/usr/bin/env python3
"""
Final Complete Chicago 311 Data Loader - ALL 4 DATABASES WORKING
Fixes NumPy 2.0 and DuckDB column type issues
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

class FinalAll4DatabasesLoader:
    """Final loader that gets ALL 4 databases working with 12.4M records."""
    
    def __init__(self):
        self.api_url = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"
        self.batch_size = 5000
        self.max_retries = 3
        self.delay_between_batches = 0.5
        
        # Database handlers
        self.handlers = {}
        
        # Progress tracking
        self.total_processed = 0
        self.start_time = None
        
    def setup_databases(self):
        """Setup all 4 databases with complete fixes."""
        logger.info("🔧 Setting up ALL 4 databases with complete fixes...")
        
        # 1. MongoDB (working fine)
        try:
            logger.info("Setting up MongoDB...")
            mongo_handler = MongoDBHandler()
            mongo_handler.collection.delete_many({})
            self.handlers['MongoDB'] = mongo_handler
            logger.info("✅ MongoDB ready")
        except Exception as e:
            logger.error(f"❌ MongoDB setup failed: {e}")
            
        # 2. Elasticsearch with COMPLETE NumPy 2.0 fix
        try:
            logger.info("Setting up Elasticsearch with complete NumPy 2.0 fix...")
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
            
        # 3. PostgreSQL (working fine)
        try:
            logger.info("Setting up PostgreSQL...")
            import psycopg2
            
            class WorkingPostgreSQLHandler:
                def __init__(self):
                    try:
                        self.connection = psycopg2.connect(
                            host="localhost",
                            port=5432,
                            database="postgres",
                            user="postgres",
                            password="postgres"
                        )
                        self.connection.autocommit = True
                    except:
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
                    """
                    
                    with self.connection.cursor() as cursor:
                        cursor.execute(create_sql)
                
                def bulk_insert(self, data):
                    if not data:
                        return 0
                    
                    clean_data = []
                    for record in data:
                        clean_record = {}
                        for key, value in record.items():
                            if key == '_id':
                                continue
                            if isinstance(value, str) and len(value) > 500:
                                clean_record[key] = value[:500]
                            elif pd.isna(value) or value == '' or value == 'nan':
                                clean_record[key] = None
                            else:
                                clean_record[key] = value
                        clean_data.append(clean_record)
                    
                    if not clean_data:
                        return 0
                    
                    available_columns = list(clean_data[0].keys())
                    table_columns = [
                        'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 'status',
                        'created_date', 'last_modified_date', 'closed_date', 'street_address',
                        'city', 'state', 'zip_code', 'street_number', 'street_direction',
                        'street_name', 'street_type', 'duplicate_ssr', 'legacy_sr_number',
                        'legacy_record', 'parent_sr_number', 'community_area', 'ward',
                        'electrical_district', 'electricity_grid', 'police_sector',
                        'police_district', 'police_beat', 'precinct',
                        'sanitation_division_days', 'created_hour', 'created_day_of_week',
                        'created_month', 'x_coordinate', 'y_coordinate', 'latitude',
                        'longitude', 'location_point', 'origin', 'type_of_service_request',
                        'most_recent_action', 'current_activity', 'number_of_days_pending',
                        'completion_date', 'due_date', 'ssa', 'channel',
                        'historical_wards_03_15', 'zip_codes', 'community_areas',
                        'census_tracts', 'wards'
                    ]
                    
                    insert_columns = [col for col in table_columns if col in available_columns]
                    placeholders = ', '.join(['%s'] * len(insert_columns))
                    columns_str = ', '.join(insert_columns)
                    
                    insert_sql = f"""
                    INSERT INTO chicago_311_requests ({columns_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (sr_number) DO NOTHING
                    """
                    
                    values = []
                    for record in clean_data:
                        row_values = [record.get(col) for col in insert_columns]
                        values.append(tuple(row_values))
                    
                    try:
                        with self.connection.cursor() as cursor:
                            cursor.executemany(insert_sql, values)
                            return cursor.rowcount
                    except Exception as e:
                        logger.error(f"PostgreSQL insert error: {e}")
                        return 0
            
            pg_handler = WorkingPostgreSQLHandler()
            self.handlers['PostgreSQL'] = pg_handler
            logger.info("✅ PostgreSQL ready")
        except Exception as e:
            logger.error(f"❌ PostgreSQL setup failed: {e}")
            
        # 4. DuckDB with PROPER column types (no timestamp confusion)
        try:
            logger.info("Setting up DuckDB with proper column types...")
            import duckdb
            
            class WorkingDuckDBHandler:
                def __init__(self):
                    self.db_path = "chicago_311.duckdb"
                    self.conn = duckdb.connect(self.db_path)
                    self.create_table()
                
                def create_table(self):
                    # All VARCHAR to avoid type conversion issues
                    create_sql = """
                    DROP TABLE IF EXISTS chicago_311_requests;
                    CREATE TABLE chicago_311_requests (
                        sr_number VARCHAR,
                        sr_type VARCHAR,
                        sr_short_code VARCHAR,
                        owner_department VARCHAR,
                        status VARCHAR,
                        created_date VARCHAR,
                        last_modified_date VARCHAR,
                        closed_date VARCHAR,
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
                        community_area VARCHAR,
                        ward VARCHAR,
                        electrical_district VARCHAR,
                        electricity_grid VARCHAR,
                        police_sector VARCHAR,
                        police_district VARCHAR,
                        police_beat VARCHAR,
                        precinct VARCHAR,
                        sanitation_division_days VARCHAR,
                        created_hour VARCHAR,
                        created_day_of_week VARCHAR,
                        created_month VARCHAR,
                        x_coordinate VARCHAR,
                        y_coordinate VARCHAR,
                        latitude VARCHAR,
                        longitude VARCHAR,
                        location_point VARCHAR,
                        origin VARCHAR,
                        type_of_service_request VARCHAR,
                        most_recent_action VARCHAR,
                        current_activity VARCHAR,
                        number_of_days_pending VARCHAR,
                        completion_date VARCHAR,
                        due_date VARCHAR,
                        ssa VARCHAR,
                        channel VARCHAR,
                        historical_wards_03_15 VARCHAR,
                        zip_codes VARCHAR,
                        community_areas VARCHAR,
                        census_tracts VARCHAR,
                        wards VARCHAR
                    );
                    """
                    self.conn.execute(create_sql)
                
                def bulk_insert(self, data):
                    if not data:
                        return 0
                    
                    # Clean data and convert everything to strings
                    clean_data = []
                    for record in data:
                        clean_record = {}
                        for key, value in record.items():
                            if key == '_id':
                                continue
                            # Convert everything to string to avoid type issues
                            if pd.isna(value) or value is None:
                                clean_record[key] = None
                            else:
                                clean_record[key] = str(value)
                        clean_data.append(clean_record)
                    
                    df = pd.DataFrame(clean_data)
                    
                    try:
                        self.conn.register('temp_data', df)
                        self.conn.execute("INSERT INTO chicago_311_requests SELECT * FROM temp_data")
                        return len(df)
                    except Exception as e:
                        logger.error(f"DuckDB insert error: {e}")
                        return 0
            
            duckdb_handler = WorkingDuckDBHandler()
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
                    return []
                
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
            
            # Basic fields
            string_fields = [
                'sr_number', 'sr_type', 'sr_short_code', 'owner_department', 'status',
                'origin', 'type_of_service_request', 'most_recent_action', 'current_activity',
                'ssa', 'channel', 'street_address', 'city', 'state', 'zip_code',
                'street_number', 'street_direction', 'street_name', 'street_type',
                'duplicate_ssr', 'legacy_sr_number', 'legacy_record', 'parent_sr_number',
                'electrical_district', 'electricity_grid', 'police_sector', 'police_district',
                'police_beat', 'precinct', 'sanitation_division_days', 'created_day_of_week'
            ]
            
            for field in string_fields:
                value = record.get(field)
                if value is None or value == '':
                    transformed[field] = None
                elif isinstance(value, str):
                    transformed[field] = value.strip()
                else:
                    transformed[field] = str(value) if value else None
            
            # Date fields
            date_fields = ['created_date', 'last_modified_date', 'closed_date', 'completion_date', 'due_date']
            for field in date_fields:
                if field in record and record[field]:
                    try:
                        transformed[field] = pd.to_datetime(record[field], errors='coerce')
                    except:
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Numeric fields
            int_fields = ['community_area', 'ward', 'created_hour', 'created_month', 'number_of_days_pending']
            for field in int_fields:
                if field in record and record[field] is not None:
                    try:
                        transformed[field] = int(float(record[field])) if record[field] else None
                    except (ValueError, TypeError):
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Float fields
            float_fields = ['x_coordinate', 'y_coordinate', 'latitude', 'longitude']
            for field in float_fields:
                if field in record and record[field] is not None:
                    try:
                        transformed[field] = float(record[field]) if record[field] else None
                    except (ValueError, TypeError):
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Computed region fields
            computed_fields = {
                'historical_wards_03_15': ':@computed_region_rpca_8um6',
                'zip_codes': ':@computed_region_6mkv_f3dw',
                'community_areas': ':@computed_region_vrxf_vc4k',
                'census_tracts': ':@computed_region_bdys_3d7i',
                'wards': ':@computed_region_43wa_7qmu'
            }
            
            for target_field, source_field in computed_fields.items():
                if source_field in record and record[source_field] is not None:
                    try:
                        value = record[source_field]
                        transformed[target_field] = int(float(value)) if value else None
                    except (ValueError, TypeError):
                        transformed[target_field] = None
                else:
                    transformed[target_field] = None
            
            # Handle location data
            if 'location' in record and record['location']:
                location = record['location']
                if isinstance(location, dict) and 'coordinates' in location:
                    coords = location['coordinates']
                    if len(coords) >= 2:
                        transformed['longitude'] = float(coords[0])
                        transformed['latitude'] = float(coords[1])
                        transformed['location_point'] = f"POINT({coords[0]} {coords[1]})"
            
            if not transformed.get('location_point') and transformed.get('latitude') and transformed.get('longitude'):
                transformed['location_point'] = f"POINT({transformed['longitude']} {transformed['latitude']})"
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            return {}
    
    def insert_to_database(self, db_name: str, data: List[Dict]) -> int:
        """Insert data into specific database with ALL FIXES."""
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
                            # COMPLETE NumPy 2.0 fix
                            if value is None or pd.isna(value):
                                clean_doc[key] = None
                            elif isinstance(value, np.floating):
                                clean_doc[key] = float(value)
                            elif isinstance(value, np.integer):  
                                clean_doc[key] = int(value)
                            elif isinstance(value, np.bool_):
                                clean_doc[key] = bool(value)
                            elif isinstance(value, np.ndarray):
                                clean_doc[key] = value.tolist()
                            elif hasattr(value, 'item'):  # Any other numpy type
                                clean_doc[key] = value.item()
                            elif str(type(value)).startswith("<class 'numpy."):  # Catch any numpy type
                                if hasattr(value, 'item'):
                                    clean_doc[key] = value.item()
                                else:
                                    clean_doc[key] = str(value)
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
                        if inserted_count > 0:
                            logger.info(f"✅ {db_name}: {inserted_count} records")
                        else:
                            logger.warning(f"⚠️  {db_name}: 0 records")
                    except Exception as e:
                        logger.error(f"❌ {db_name}: {e}")
                        results[db_name] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            return {}
    
    def run_complete_loading(self):
        """Load complete 12.4M dataset into ALL 4 databases."""
        target_records = 12400000
        logger.info(f"🚀 FINAL COMPLETE LOADING - ALL 4 DATABASES!")
        logger.info(f"📊 Target: {target_records:,} records")
        
        # Setup databases
        self.setup_databases()
        if len(self.handlers) != 4:
            logger.error(f"❌ Only {len(self.handlers)} databases ready! Need all 4!")
            logger.info(f"Available: {list(self.handlers.keys())}")
            # Continue anyway with available databases
        
        self.start_time = time.time()
        
        # Progress tracking
        last_progress_time = time.time()
        progress_interval = 60  # Report every minute
        
        offset = 0
        db_totals = {db_name: 0 for db_name in self.handlers.keys()}
        consecutive_failures = 0
        
        try:
            while offset < target_records:
                # Process batch
                batch_results = self.process_batch(offset)
                
                if not batch_results or all(count == 0 for count in batch_results.values()):
                    consecutive_failures += 1
                    logger.warning(f"❌ No data or all failed (failure #{consecutive_failures})")
                    if consecutive_failures >= 5:
                        logger.error("❌ Too many consecutive failures, stopping...")
                        break
                else:
                    consecutive_failures = 0
                
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
                    total_loaded = 0
                    for db_name, count in db_totals.items():
                        logger.info(f"   {db_name}: {count:,} records")
                        total_loaded += count
                    logger.info(f"🎯 Total across all DBs: {total_loaded:,}")
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
        logger.info("🎉 FINAL COMPLETE LOADING FINISHED!")
        logger.info(f"⏱️  Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"📊 Total processed: {self.total_processed:,} records")
        logger.info(f"⚡ Average rate: {avg_rate:.0f} records/sec")
        logger.info("💾 Final database counts:")
        
        total_loaded = 0
        for db_name, count in db_totals.items():
            logger.info(f"   {db_name}: {count:,} records")
            total_loaded += count
        
        logger.info(f"🎯 Total loaded across ALL 4 databases: {total_loaded:,}")
        
        # Check if all 4 databases have data
        successful_dbs = [db for db, count in db_totals.items() if count > 0]
        if len(successful_dbs) == 4:
            logger.info("🎉 SUCCESS! All 4 databases loaded successfully!")
        else:
            logger.warning(f"⚠️  Only {len(successful_dbs)} out of 4 databases loaded: {successful_dbs}")
        
        logger.info("=" * 80)
        
        return db_totals

def main():
    """Main execution."""
    try:
        loader = FinalAll4DatabasesLoader()
        results = loader.run_complete_loading()
        
        # Check if ready for benchmarking
        successful_dbs = [db for db, count in results.items() if count > 0]
        if len(successful_dbs) >= 2:
            logger.info(f"🚀 Ready for comprehensive benchmarking on {len(successful_dbs)} databases!")
        else:
            logger.error("❌ Not enough databases loaded for benchmarking")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Loading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()