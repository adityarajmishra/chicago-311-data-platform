#!/usr/bin/env python3
"""
Local PostgreSQL Database Handler for Chicago 311 Data Platform
Creates and manages a local PostgreSQL database
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import time
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalPostgreSQLHandler:
    """Handle local PostgreSQL database operations for Chicago 311 data."""
    
    def __init__(self, database: str = "chicago_311_local", user: str = "postgres", 
                 password: str = "postgres", host: str = "localhost", port: int = 5432):
        """Initialize local PostgreSQL connection."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
        self.setup_local_database()
        self.connect()
    
    def setup_local_database(self):
        """Setup local PostgreSQL database if it doesn't exist."""
        try:
            # First try to connect to default postgres database to create our database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database="postgres",
                user=self.user,
                password=self.password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.database}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {self.database}")
                logger.info(f"✅ Created database: {self.database}")
            else:
                logger.info(f"✅ Database {self.database} already exists")
            
            cursor.close()
            conn.close()
            
        except psycopg2.Error as e:
            logger.warning(f"⚠️ Could not setup local PostgreSQL: {e}")
            logger.info("📝 To use PostgreSQL, please install and start PostgreSQL locally:")
            logger.info("   brew install postgresql")
            logger.info("   brew services start postgresql")
            logger.info("   createuser -s postgres")
            raise
    
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
            logger.info(f"✅ Connected to local PostgreSQL: {self.database}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to local PostgreSQL: {e}")
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
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> tuple:
        """Execute a query and return results."""
        start_time = time.time()
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.4f} seconds, returned {len(results)} rows")
            return results, execution_time
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("✅ Local PostgreSQL connection closed")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()