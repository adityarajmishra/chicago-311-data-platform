#!/usr/bin/env python3
"""
Chicago 311 Data Pipeline
Main script to extract data from Chicago Data Portal and load into MongoDB and Elasticsearch
"""
import logging
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add src to path
sys.path.append('/Users/rahulmishra/Desktop/MS-Data_Science/Group Project/chicago-311-data-platform')

from src.data_extraction.chicago_data_extractor import ChicagoDataExtractor, validate_api_connection
from src.databases.mongodb_handler import MongoDBHandler, test_mongodb_connection
from src.databases.elasticsearch_handler import ElasticsearchHandler, test_elasticsearch_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChicagoDataPipeline:
    """Main pipeline for Chicago 311 data processing."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.extractor = None
        self.mongo_handler = None
        self.es_handler = None
        self.stats = {
            'total_extracted': 0,
            'mongo_inserted': 0,
            'es_indexed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def initialize_connections(self) -> bool:
        """Initialize all database connections."""
        logger.info("🔗 Initializing connections...")
        
        try:
            # Test API connection
            if not validate_api_connection():
                logger.error("❌ Chicago Data Portal API connection failed")
                return False
            
            # Test MongoDB connection
            if not test_mongodb_connection():
                logger.error("❌ MongoDB connection failed")
                return False
            
            # Test Elasticsearch connection
            if not test_elasticsearch_connection():
                logger.error("❌ Elasticsearch connection failed")
                return False
            
            # Initialize handlers
            self.extractor = ChicagoDataExtractor()
            self.mongo_handler = MongoDBHandler()
            self.es_handler = ElasticsearchHandler()
            
            logger.info("✅ All connections initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing connections: {e}")
            return False
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a single batch of data."""
        if not batch_data:
            return {'mongo_result': {'inserted': 0, 'errors': 0}, 
                   'es_result': {'indexed': 0, 'errors': 0}}
        
        batch_stats = {'mongo_result': None, 'es_result': None}
        
        # Insert into MongoDB
        try:
            mongo_result = self.mongo_handler.insert_batch(batch_data)
            batch_stats['mongo_result'] = mongo_result
            logger.info(f"📊 MongoDB: {mongo_result['inserted']} inserted, {mongo_result['errors']} errors")
        except Exception as e:
            logger.error(f"❌ MongoDB batch processing error: {e}")
            batch_stats['mongo_result'] = {'inserted': 0, 'errors': len(batch_data)}
        
        # Index in Elasticsearch
        try:
            es_result = self.es_handler.bulk_index(batch_data)
            batch_stats['es_result'] = es_result
            logger.info(f"🔍 Elasticsearch: {es_result['indexed']} indexed, {es_result['errors']} errors")
        except Exception as e:
            logger.error(f"❌ Elasticsearch batch processing error: {e}")
            batch_stats['es_result'] = {'indexed': 0, 'errors': len(batch_data)}
        
        return batch_stats
    
    def run_sample_pipeline(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Run pipeline with a sample of data."""
        logger.info(f"🔬 Starting sample pipeline with {sample_size} records...")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Extract sample data
            sample_data = self.extractor.extract_sample(sample_size)
            if not sample_data:
                logger.error("❌ No sample data extracted")
                return self.stats
            
            self.stats['total_extracted'] = len(sample_data)
            logger.info(f"📥 Extracted {len(sample_data)} sample records")
            
            # Process the sample batch
            batch_stats = self.process_batch(sample_data)
            
            # Update overall stats
            self.stats['mongo_inserted'] += batch_stats['mongo_result']['inserted']
            self.stats['es_indexed'] += batch_stats['es_result']['indexed']
            self.stats['errors'] += (batch_stats['mongo_result']['errors'] + 
                                   batch_stats['es_result']['errors'])
            
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"✅ Sample pipeline completed in {duration:.2f} seconds")
            self._print_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"❌ Sample pipeline error: {e}")
            self.stats['errors'] += 1
            return self.stats
    
    def run_full_pipeline(self, max_records: int = None) -> Dict[str, Any]:
        """Run pipeline with all available data."""
        logger.info("🚀 Starting full data pipeline...")
        self.stats['start_time'] = datetime.now()
        
        try:
            processed_batches = 0
            total_processed = 0
            
            # Stream data in batches
            for batch_data in tqdm(self.extractor.stream_all_data(), desc="Processing batches"):
                if not batch_data:
                    break
                
                # Check max_records limit
                if max_records and total_processed >= max_records:
                    logger.info(f"🛑 Reached maximum record limit: {max_records}")
                    break
                
                batch_size = len(batch_data)
                self.stats['total_extracted'] += batch_size
                total_processed += batch_size
                
                # Process batch
                batch_stats = self.process_batch(batch_data)
                
                # Update stats
                self.stats['mongo_inserted'] += batch_stats['mongo_result']['inserted']
                self.stats['es_indexed'] += batch_stats['es_result']['indexed']
                self.stats['errors'] += (batch_stats['mongo_result']['errors'] + 
                                       batch_stats['es_result']['errors'])
                
                processed_batches += 1
                
                # Progress update every 10 batches
                if processed_batches % 10 == 0:
                    logger.info(f"📊 Progress: {processed_batches} batches, {total_processed:,} records processed")
                
                # Small delay to prevent overwhelming the APIs
                time.sleep(0.1)
            
            # Refresh Elasticsearch index
            self.es_handler.refresh_index()
            
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"✅ Full pipeline completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            self._print_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"❌ Full pipeline error: {e}")
            self.stats['errors'] += 1
            return self.stats
    
    def run_incremental_pipeline(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run pipeline for a specific date range."""
        logger.info(f"📅 Starting incremental pipeline from {start_date} to {end_date}")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Extract data for date range
            date_range_data = self.extractor.extract_date_range(start_date, end_date)
            if not date_range_data:
                logger.warning("⚠️ No data found for the specified date range")
                return self.stats
            
            self.stats['total_extracted'] = len(date_range_data)
            logger.info(f"📥 Extracted {len(date_range_data)} records for date range")
            
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            for i in range(0, len(date_range_data), chunk_size):
                chunk = date_range_data[i:i + chunk_size]
                batch_stats = self.process_batch(chunk)
                
                self.stats['mongo_inserted'] += batch_stats['mongo_result']['inserted']
                self.stats['es_indexed'] += batch_stats['es_result']['indexed']
                self.stats['errors'] += (batch_stats['mongo_result']['errors'] + 
                                       batch_stats['es_result']['errors'])
            
            # Refresh Elasticsearch index
            self.es_handler.refresh_index()
            
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"✅ Incremental pipeline completed in {duration:.2f} seconds")
            self._print_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"❌ Incremental pipeline error: {e}")
            self.stats['errors'] += 1
            return self.stats
    
    def _print_stats(self):
        """Print pipeline statistics."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"📥 Total Extracted: {self.stats['total_extracted']:,}")
        print(f"🗄️  MongoDB Inserted: {self.stats['mongo_inserted']:,}")
        print(f"🔍 Elasticsearch Indexed: {self.stats['es_indexed']:,}")
        print(f"❌ Total Errors: {self.stats['errors']:,}")
        
        if duration > 0:
            records_per_sec = self.stats['total_extracted'] / duration
            print(f"⚡ Processing Rate: {records_per_sec:.2f} records/second")
        
        print("="*60)
    
    def get_database_stats(self):
        """Get current database statistics."""
        logger.info("📊 Getting database statistics...")
        
        try:
            mongo_stats = self.mongo_handler.get_stats()
            es_stats = self.es_handler.get_stats()
            
            print("\n" + "="*60)
            print("📊 DATABASE STATISTICS")
            print("="*60)
            print(f"🗄️  MongoDB:")
            print(f"   📄 Total Records: {mongo_stats.get('total_records', 0):,}")
            print(f"   🗂️  Collection: {mongo_stats.get('collection_name', 'N/A')}")
            print(f"   💾 Database: {mongo_stats.get('database_name', 'N/A')}")
            
            print(f"\n🔍 Elasticsearch:")
            print(f"   📄 Total Documents: {es_stats.get('total_documents', 0):,}")
            print(f"   💾 Index Size: {es_stats.get('index_size_mb', 0)} MB")
            print(f"   🗂️  Index: {es_stats.get('index_name', 'N/A')}")
            print(f"   🔧 Shards: {es_stats.get('number_of_shards', 0)}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.extractor:
                self.extractor.close()
            if self.mongo_handler:
                self.mongo_handler.close()
            if self.es_handler:
                self.es_handler.close()
            logger.info("✅ Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Chicago 311 Data Pipeline')
    parser.add_argument('--mode', choices=['sample', 'full', 'incremental'], 
                       default='sample', help='Pipeline mode')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Sample size for sample mode')
    parser.add_argument('--max-records', type=int, help='Maximum records to process')
    parser.add_argument('--start-date', help='Start date for incremental mode (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for incremental mode (YYYY-MM-DD)')
    parser.add_argument('--stats-only', action='store_true', 
                       help='Only show database statistics')
    
    args = parser.parse_args()
    
    pipeline = ChicagoDataPipeline()
    
    try:
        # Initialize connections
        if not pipeline.initialize_connections():
            logger.error("❌ Failed to initialize pipeline connections")
            return 1
        
        # Show stats only if requested
        if args.stats_only:
            pipeline.get_database_stats()
            return 0
        
        # Run pipeline based on mode
        if args.mode == 'sample':
            pipeline.run_sample_pipeline(args.sample_size)
        elif args.mode == 'full':
            pipeline.run_full_pipeline(args.max_records)
        elif args.mode == 'incremental':
            if not args.start_date or not args.end_date:
                logger.error("❌ Incremental mode requires --start-date and --end-date")
                return 1
            pipeline.run_incremental_pipeline(args.start_date, args.end_date)
        
        # Show final database stats
        pipeline.get_database_stats()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    exit(main())