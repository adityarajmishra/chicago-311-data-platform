#!/usr/bin/env python3
"""
DuckDB Database Handler for Chicago 311 Data Platform
"""

import logging
import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDBHandler:
    """Handle DuckDB database operations for Chicago 311 data."""
    
    def __init__(self, db_path: str = "/data/chicago_311.duckdb"):
        """Initialize DuckDB connection."""
        self.db_path = db_path
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to DuckDB."""
        try:
            # Check if running in Docker container
            if os.path.exists(self.db_path):
                self.connection = duckdb.connect(self.db_path)
                logger.info(f"✅ Connected to DuckDB: {self.db_path}")
            else:
                # Fallback to in-memory database
                self.connection = duckdb.connect(":memory:")
                logger.warning(f"⚠️ DuckDB file not found, using in-memory database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            raise
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            result = self.connection.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
    
    def get_table_count(self, table_name: str) -> int:
        """Get record count from a table."""
        try:
            result = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting table count for {table_name}: {e}")
            return 0
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        try:
            result = self.connection.execute(f"DESCRIBE {table_name}").fetchall()
            return [{"column_name": row[0], "data_type": row[1], "is_nullable": row[2]} 
                    for row in result]
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return []
    
    def execute_query(self, query: str) -> tuple:
        """Execute a query and return results with timing."""
        start_time = time.time()
        try:
            result = self.connection.execute(query).fetchall()
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.4f} seconds, returned {len(result)} rows")
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.4f} seconds: {e}")
            return [], execution_time
    
    def execute_query_df(self, query: str) -> tuple:
        """Execute a query and return results as DataFrame with timing."""
        start_time = time.time()
        try:
            result = self.connection.execute(query).fetchdf()
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.4f} seconds, returned {len(result)} rows")
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.4f} seconds: {e}")
            return pd.DataFrame(), execution_time
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> List[tuple]:
        """Get sample data from a table."""
        try:
            return self.connection.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchall()
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return []
    
    def create_table(self):
        """Create the chicago_311_requests table with proper schema."""
        try:
            # Drop table if exists
            self.connection.execute("DROP TABLE IF EXISTS chicago_311_requests")
            
            # Create table with BIGINT for id to handle large INT64 values
            create_sql = """
            CREATE TABLE chicago_311_requests (
                id BIGINT,
                sr_number VARCHAR(50),
                sr_type VARCHAR(200),
                sr_short_code VARCHAR(50),
                owner_department VARCHAR(100),
                status VARCHAR(50),
                created_date TIMESTAMP,
                last_modified_date TIMESTAMP,
                closed_date TIMESTAMP,
                street_address VARCHAR(200),
                city VARCHAR(50),
                state VARCHAR(10),
                zip_code VARCHAR(20),
                street_number VARCHAR(20),
                street_direction VARCHAR(10),
                street_name VARCHAR(100),
                street_type VARCHAR(50),
                duplicate_ssr VARCHAR(10),
                legacy_sr_number VARCHAR(50),
                legacy_record VARCHAR(10),
                parent_sr_number VARCHAR(50),
                community_area INTEGER,
                ward INTEGER,
                electrical_district VARCHAR(50),
                electricity_grid VARCHAR(50),
                police_sector VARCHAR(50),
                police_district VARCHAR(50),
                police_beat VARCHAR(50),
                precinct VARCHAR(50),
                sanitation_division_days VARCHAR(50),
                created_hour INTEGER,
                created_day_of_week VARCHAR(20),
                created_month INTEGER,
                x_coordinate DOUBLE,
                y_coordinate DOUBLE,
                latitude DOUBLE,
                longitude DOUBLE,
                location_point VARCHAR(100)
            )
            """
            
            self.connection.execute(create_sql)
            logger.info("✅ DuckDB table created successfully with BIGINT ID")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create DuckDB table: {e}")
            return False
    
    def bulk_insert(self, data: List[Dict[str, Any]]) -> int:
        """Insert multiple records into the table."""
        if not data:
            return 0
            
        try:
            # Convert data to DataFrame for efficient insertion
            df = pd.DataFrame(data)
            
            # Convert problematic columns
            for col in ['created_date', 'last_modified_date', 'closed_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Insert data
            self.connection.execute("INSERT INTO chicago_311_requests SELECT * FROM df")
            
            inserted_count = len(data)
            logger.info(f"✅ Inserted {inserted_count} records into DuckDB")
            return inserted_count
        except Exception as e:
            logger.error(f"❌ Failed to bulk insert into DuckDB: {e}")
            return 0

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("✅ DuckDB connection closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()