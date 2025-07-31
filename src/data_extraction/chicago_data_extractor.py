"""Chicago 311 data extraction module."""
import logging
import time
from typing import List, Dict, Any, Optional
from sodapy import Socrata
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoDataExtractor:
    """Extract data from Chicago Data Portal."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.domain = os.getenv('CHICAGO_API_DOMAIN', 'data.cityofchicago.org')
        self.dataset_id = os.getenv('DATASET_ID', 'v6vf-nfxy')
        self.app_token = os.getenv('CHICAGO_API_TOKEN')
        
        if not self.app_token:
            logger.warning("No API token provided. Rate limiting may apply.")
            logger.info("To get a free API token, visit: https://dev.socrata.com/foundry/data.cityofchicago.org/v6vf-nfxy")
        
        # Use None for app_token if empty string
        token = self.app_token if self.app_token else None
        self.client = Socrata(self.domain, token)
        self.batch_size = int(os.getenv('BATCH_SIZE', 10000))
    
    def get_total_records(self) -> int:
        """Get total number of records in the dataset."""
        try:
            # Get metadata to find total records
            metadata = self.client.get_metadata(self.dataset_id)
            total_rows = metadata.get('rowsUpdatedAt', 0)
            logger.info(f"Total records in dataset: {total_rows:,}")
            return total_rows
        except Exception as e:
            logger.error(f"Error getting total records: {e}")
            return 0
    
    def extract_batch(self, offset: int = 0, limit: int = None) -> List[Dict[str, Any]]:
        """Extract a batch of data with proper pagination handling."""
        if limit is None:
            limit = self.batch_size
        
        try:
            logger.info(f"Extracting batch: offset={offset:,}, limit={limit:,}")
            
            # Use $offset and $limit parameters for proper pagination
            # This is the correct way to paginate through large datasets
            results = self.client.get(
                self.dataset_id,
                limit=limit,
                offset=offset,
                order="created_date DESC"
            )
            
            logger.info(f"Successfully extracted {len(results)} records")
            return results
            
        except Exception as e:
            logger.error(f"Error extracting batch at offset {offset}: {e}")
            # Implement retry logic for network issues
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                logger.info("Retrying due to network issue...")
                time.sleep(2)
                try:
                    results = self.client.get(
                        self.dataset_id,
                        limit=limit,
                        offset=offset,
                        order="created_date DESC"
                    )
                    logger.info(f"Retry successful: extracted {len(results)} records")
                    return results
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
            return []
    
    def extract_sample(self, sample_size: int = 1000) -> List[Dict[str, Any]]:
        """Extract a sample of data for testing."""
        logger.info(f"Extracting sample of {sample_size} records")
        return self.extract_batch(offset=0, limit=sample_size)
    
    def extract_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Extract data for a specific date range."""
        try:
            logger.info(f"Extracting data from {start_date} to {end_date}")
            
            where_clause = f"created_date between '{start_date}' and '{end_date}'"
            results = self.client.get(
                self.dataset_id,
                where=where_clause,
                limit=50000,  # Adjust based on expected volume
                order="created_date DESC"
            )
            
            logger.info(f"Extracted {len(results)} records for date range")
            return results
            
        except Exception as e:
            logger.error(f"Error extracting date range data: {e}")
            return []
    
    def extract_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Extract data filtered by status."""
        try:
            logger.info(f"Extracting data with status: {status}")
            
            where_clause = f"status = '{status}'"
            results = self.client.get(
                self.dataset_id,
                where=where_clause,
                limit=50000,
                order="created_date DESC"
            )
            
            logger.info(f"Extracted {len(results)} records with status {status}")
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data by status: {e}")
            return []
    
    def stream_all_data(self):
        """Generator to stream all data in batches."""
        offset = 0
        total_processed = 0
        
        logger.info("Starting to stream all data...")
        
        while True:
            batch = self.extract_batch(offset=offset)
            
            if not batch or len(batch) == 0:
                logger.info("No more data to extract")
                break
            
            total_processed += len(batch)
            logger.info(f"Processed {total_processed:,} records so far...")
            
            yield batch
            
            # If we got less than batch_size, we're at the end
            if len(batch) < self.batch_size:
                break
            
            offset += self.batch_size
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)
        
        logger.info(f"Streaming completed. Total records processed: {total_processed:,}")
    
    def close(self):
        """Close the Socrata client."""
        if hasattr(self.client, 'close'):
            self.client.close()
        logger.info("Data extractor closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Utility functions
def validate_api_connection() -> bool:
    """Validate connection to Chicago Data Portal API."""
    try:
        extractor = ChicagoDataExtractor()
        test_data = extractor.extract_batch(offset=0, limit=1)
        extractor.close()
        
        if test_data and len(test_data) > 0:
            logger.info("✅ API connection validated successfully")
            return True
        else:
            logger.error("❌ API connection validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ API connection validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test the extractor
    if validate_api_connection():
        with ChicagoDataExtractor() as extractor:
            sample_data = extractor.extract_sample(10)
            print(f"Sample data extracted: {len(sample_data)} records")
            if sample_data:
                print("Sample record keys:", sample_data[0].keys())