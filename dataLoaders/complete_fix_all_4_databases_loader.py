#!/usr/bin/env python3
"""
Complete Fix - ALL 4 Databases Working Chicago 311 Data Loader
FIXES the persistent Elasticsearch NumPy 2.0 issue by avoiding NumPy types entirely
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

class CompleteFixAll4DatabasesLoader:
    """Complete fix that gets ALL 4 databases working with 12.4M records."""
    
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
        logger.info("🔧 Setting up ALL 4 databases with COMPLETE FIXES...")
        
        # 1. MongoDB (already working)
        try:
            logger.info("Setting up MongoDB...")
            mongo_handler = MongoDBHandler()
            mongo_handler.collection.delete_many({})
            self.handlers['MongoDB'] = mongo_handler
            logger.info("✅ MongoDB ready")
        except Exception as e:
            logger.error(f"❌ MongoDB setup failed: {e}")
            
        # 2. Elasticsearch with COMPLETE NumPy fix - avoid pandas entirely
        try:
            logger.info("Setting up Elasticsearch with COMPLETE NumPy elimination...")
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
            
        # 3. PostgreSQL (already working)
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
                            elif value is None or value == '' or str(value) == 'nan':
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
            
        # 4. DuckDB (already working)
        try:
            logger.info("Setting up DuckDB...")
            import duckdb
            
            class WorkingDuckDBHandler:
                def __init__(self):
                    self.db_path = "chicago_311.duckdb"
                    self.conn = duckdb.connect(self.db_path)
                    self.create_table()
                
                def create_table(self):
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
                    
                    # Convert everything to strings to avoid type issues
                    clean_data = []
                    for record in data:
                        clean_record = {}
                        for key, value in record.items():
                            if key == '_id':
                                continue
                            if value is None or str(value) == 'nan':
                                clean_record[key] = None
                            else:
                                clean_record[key] = str(value)
                        clean_data.append(clean_record)
                    
                    # Convert to DataFrame with explicit string conversion
                    try:
                        df = pd.DataFrame(clean_data)
                        # Ensure all columns are strings
                        for col in df.columns:
                            df[col] = df[col].astype(str).replace({'nan': None, 'None': None})
                        
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
    
    def clean_value_for_databases(self, value: Any) -> Any:
        """Clean value to avoid NumPy and pandas issues."""
        if value is None:
            return None
        
        # Convert pandas/numpy types to native Python types
        if hasattr(value, 'item'):  # NumPy scalar
            return value.item()
        
        # Handle pandas Series, DataFrames, etc.
        if hasattr(value, 'dtype'):
            if hasattr(value, 'item'):
                return value.item()
            else:
                return str(value)
        
        # Handle datetime objects
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        
        # Convert NumPy arrays to lists
        if hasattr(value, 'tolist'):
            return value.tolist()
        
        # Handle string representations of NaN
        if str(value) in ['nan', 'NaT', 'None']:
            return None
        
        return value
    
    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API record for database insertion - AVOID PANDAS/NUMPY."""
        try:
            transformed = {}
            
            # Basic fields - use native Python types only
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
                else:
                    transformed[field] = str(value).strip()
            
            # Date fields - use datetime directly, avoid pandas
            date_fields = ['created_date', 'last_modified_date', 'closed_date', 'completion_date', 'due_date']
            for field in date_fields:
                if field in record and record[field]:
                    try:
                        # Use datetime.fromisoformat or strptime instead of pandas
                        date_str = str(record[field])
                        if 'T' in date_str:
                            # ISO format
                            transformed[field] = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            # Try common formats
                            try:
                                transformed[field] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                            except:
                                transformed[field] = datetime.strptime(date_str[:19], '%Y-%m-%dT%H:%M:%S')
                    except Exception as e:
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Numeric fields - use native Python types
            int_fields = ['community_area', 'ward', 'created_hour', 'created_month', 'number_of_days_pending']
            for field in int_fields:
                if field in record and record[field] is not None:
                    try:
                        value = record[field]
                        if str(value) in ['', 'nan', 'None']:
                            transformed[field] = None
                        else:
                            transformed[field] = int(float(value))
                    except (ValueError, TypeError):
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Float fields - use native Python types
            float_fields = ['x_coordinate', 'y_coordinate', 'latitude', 'longitude']
            for field in float_fields:
                if field in record and record[field] is not None:
                    try:
                        value = record[field]
                        if str(value) in ['', 'nan', 'None']:
                            transformed[field] = None
                        else:
                            transformed[field] = float(value)
                    except (ValueError, TypeError):
                        transformed[field] = None
                else:
                    transformed[field] = None
            
            # Computed region fields - use native Python types
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
                        if str(value) in ['', 'nan', 'None']:
                            transformed[target_field] = None
                        else:
                            transformed[target_field] = int(float(value))
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
                    # Clean data for MongoDB
                    clean_data = []
                    for doc in data:
                        clean_doc = {}
                        for key, value in doc.items():
                            clean_doc[key] = self.clean_value_for_databases(value)
                        clean_data.append(clean_doc)
                    
                    result = handler.collection.insert_many(clean_data, ordered=False)
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
                            # COMPLETELY avoid NumPy - clean to native Python types
                            cleaned_value = self.clean_value_for_databases(value)
                            
                            # Final safety check - convert any remaining problematic types
                            if cleaned_value is None:
                                clean_doc[key] = None
                            elif isinstance(cleaned_value, (int, float, str, bool, list, dict)):
                                clean_doc[key] = cleaned_value
                            elif isinstance(cleaned_value, datetime):
                                clean_doc[key] = cleaned_value.isoformat()
                            else:
                                # Convert anything else to string as last resort
                                clean_doc[key] = str(cleaned_value)
                        
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
                # Clean data for SQL databases
                clean_data = []
                for doc in data:
                    clean_doc = {}
                    for key, value in doc.items():
                        clean_doc[key] = self.clean_value_for_databases(value)
                    clean_data.append(clean_doc)
                
                return handler.bulk_insert(clean_data)
            
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
            
            # Transform data - avoid pandas/numpy here
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
        logger.info(f"🚀 COMPLETE FIX - ALL 4 DATABASES LOADING!")
        logger.info(f"📊 Target: {target_records:,} records")
        
        # Setup databases
        self.setup_databases()
        if len(self.handlers) < 4:
            logger.warning(f"⚠️ Only {len(self.handlers)} databases ready! Should be 4!")
            logger.info(f"Available: {list(self.handlers.keys())}")
        
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
                    working_databases = 0
                    for db_name, count in db_totals.items():
                        if count > 0:
                            logger.info(f"   ✅ {db_name}: {count:,} records")
                            working_databases += 1
                        else:
                            logger.info(f"   ❌ {db_name}: {count:,} records")
                        total_loaded += count
                    logger.info(f"🎯 Total across all DBs: {total_loaded:,}")
                    logger.info(f"🔥 Working databases: {working_databases}/4")
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
        logger.info("🎉 COMPLETE FIX LOADING FINISHED!")
        logger.info(f"⏱️  Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"📊 Total processed: {self.total_processed:,} records")
        logger.info(f"⚡ Average rate: {avg_rate:.0f} records/sec")
        logger.info("💾 Final database counts:")
        
        total_loaded = 0
        successful_dbs = 0
        for db_name, count in db_totals.items():
            if count > 0:
                logger.info(f"   ✅ {db_name}: {count:,} records")
                successful_dbs += 1
            else:
                logger.info(f"   ❌ {db_name}: {count:,} records")
            total_loaded += count
        
        logger.info(f"🎯 Total loaded across ALL databases: {total_loaded:,}")
        logger.info(f"🔥 Successful databases: {successful_dbs}/4")
        
        # Check if ALL 4 databases have data
        if successful_dbs == 4:
            logger.info("🎉🎉 SUCCESS! ALL 4 DATABASES loaded successfully!")
        else:
            logger.warning(f"⚠️  Only {successful_dbs} out of 4 databases loaded successfully")
        
        logger.info("=" * 80)
        
        return db_totals

def main():
    """Main execution."""
    try:
        loader = CompleteFixAll4DatabasesLoader()
        results = loader.run_complete_loading()
        
        # Check if ready for benchmarking
        successful_dbs = [db for db, count in results.items() if count > 0]
        if len(successful_dbs) == 4:
            logger.info("🚀🚀 ALL 4 DATABASES READY FOR COMPREHENSIVE BENCHMARKING!")
        else:
            logger.info(f"🚀 {len(successful_dbs)} databases ready for benchmarking")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Loading interrupted by user")
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()