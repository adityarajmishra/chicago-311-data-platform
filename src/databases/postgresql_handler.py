#!/usr/bin/env python3
"""
PostgreSQL Database Handler for Chicago 311 Data Platform
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLHandler:
    """Handle PostgreSQL database operations for Chicago 311 data."""
    
    def __init__(self, host: str = "192.168.254.105", port: int = 5437, 
                 database: str = "chicago_311_star", user: str = "postgres", 
                 password: str = "sql"):
        """Initialize PostgreSQL connection."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
        self.connect()
    
    def connect(self):
        """Establish connection to PostgreSQL."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.database} at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """Get a database cursor with automatic cleanup."""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def get_table_count(self, table_name: str) -> int:
        """Get record count from a table."""
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            return cursor.fetchall()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a query and return results."""
        start_time = time.time()
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.4f} seconds, returned {len(results)} rows")
            return results, execution_time
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample data from a table."""
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT %s", (limit,))
            return cursor.fetchall()
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("✅ PostgreSQL connection closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()