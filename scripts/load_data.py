"""Script to load Chicago 311 data into databases."""
import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_extraction.chicago_data_extractor import ChicagoDataExtractor
from src.data_extraction.data_validator import DataValidator
from src.databases.mongodb_handler import MongoDBHandler
from src.databases.elasticsearch_handler import ElasticsearchHandler
from src.utils.helpers import setup_logging, progress_bar

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load Chicago 311 data')
    
    parser.add_argument('--sample', action='store_true', 
                       help='Load sample data (1000 records)')
    parser.add_argument('--full', action='store_true', 
                       help='Load full dataset (12M+ records)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for processing')
    parser.add_argument('--mongodb-only', action='store_true',
                       help='Load data only to MongoDB')
    parser.add_argument('--elasticsearch-only', action='store_true',
                       help='Load data only to Elasticsearch')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate data before loading')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    return parser.parse_args()

class DataLoader:
    """Data loading orchestrator."""
    
    def __init__(self, args):
        """Initialize data loader."""
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.extractor = ChicagoDataExtractor()
        self.validator = DataValidator() if args.validate else None
        
        # Initialize database handlers
        self.mongo_handler = None
        self.es_handler = None
        
        if not args.elasticsearch_only:
            try:
                self.mongo_handler = MongoDBHandler()
                self.logger.info("✅ MongoDB handler initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize MongoDB: {e}")
                if not args.mongodb_only:
                    self.logger.info("Continuing with Elasticsearch only...")
        
        if not args.mongodb_only:
            try:
                self.es_handler = ElasticsearchHandler()
                self.logger.info("✅ Elasticsearch handler initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Elasticsearch: {e}")
                if not args.elasticsearch_only:
                    self.logger.info("Continuing with MongoDB only...")
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_validated': 0,
            'total_mongo_inserted': 0,
            'total_es_indexed': 0,
            'validation_errors': 0,
            'mongo_errors': 0,
            'es_errors': 0,
            'start_time': time.time()
        }
    
    def load_sample_data(self) -> None:
        """Load sample data for testing."""
        self.logger.info("📊 Loading sample data (1000 records)...")
        
        try:
            # Extract sample data
            sample_data = self.extractor.extract_sample(1000)
            if not sample_data:
                self.logger.error("❌ No sample data extracted")
                return
            
            self.logger.info(f"Extracted {len(sample_data)} sample records")
            
            # Process the batch
            self._process_batch(sample_data)
            
            # Print results
            self._print_summary()
            
        except Exception as e:
            self.logger.error(f"❌ Error loading sample data: {e}")
    
    def load_full_data(self) -> None:
        """Load full dataset."""
        self.logger.info("📊 Loading full dataset (this may take several hours)...")
        
        try:
            batch_count = 0
            
            # Stream data in batches
            for batch in self.extractor.stream_all_data():
                batch_count += 1
                self.logger.info(f"Processing batch #{batch_count} ({len(batch)} records)")
                
                # Process the batch
                self._process_batch(batch)
                
                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    self._print_progress()
                
                # Optional: Add delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Print final results
            self._print_summary()
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Loading interrupted by user")
            self._print_summary()
        except Exception as e:
            self.logger.error(f"❌ Error loading full data: {e}")
            self._print_summary()
    
    def _process_batch(self, batch_data: list) -> None:
            """Process a batch of data."""
            if not batch_data:
                return
            
            self.stats['total_processed'] += len(batch_data)
            
            # Validate data if enabled
            validated_data = batch_data
            if self.validator:
                validation_results = self.validator.validate_batch(batch_data)
                validated_data = validation_results['valid_data']
                
                self.stats['total_validated'] += validation_results['valid_records']
                self.stats['validation_errors'] += validation_results['invalid_records']
                
                if validation_results['invalid_records'] > 0:
                    self.logger.warning(f"⚠️ {validation_results['invalid_records']} invalid records in batch")
            
            if not validated_data:
                self.logger.warning("⚠️ No valid data in batch after validation")
                return
            
            # Load to MongoDB
            if self.mongo_handler:
                try:
                    mongo_result = self.mongo_handler.insert_batch(validated_data)
                    self.stats['total_mongo_inserted'] += mongo_result['inserted']
                    self.stats['mongo_errors'] += mongo_result['errors']
                    
                    if mongo_result['errors'] > 0:
                        self.logger.warning(f"⚠️ MongoDB: {mongo_result['errors']} errors in batch")
                        
                except Exception as e:
                    self.logger.error(f"❌ MongoDB batch insert error: {e}")
                    self.stats['mongo_errors'] += len(validated_data)
            
            # Load to Elasticsearch
            if self.es_handler:
                try:
                    es_result = self.es_handler.bulk_index(validated_data)
                    self.stats['total_es_indexed'] += es_result['indexed']
                    self.stats['es_errors'] += es_result['errors']
                    
                    if es_result['errors'] > 0:
                        self.logger.warning(f"⚠️ Elasticsearch: {es_result['errors']} errors in batch")
                        
                except Exception as e:
                    self.logger.error(f"❌ Elasticsearch batch index error: {e}")
                    self.stats['es_errors'] += len(validated_data)
        
            def _print_progress(self) -> None:
                """Print current progress."""
                elapsed_time = time.time() - self.stats['start_time']
                rate = self.stats['total_processed'] / elapsed_time if elapsed_time > 0 else 0
                
                self.logger.info("📈 Progress Update:")
                self.logger.info(f"   Processed: {self.stats['total_processed']:,} records")
                self.logger.info(f"   Rate: {rate:.1f} records/second")
                self.logger.info(f"   MongoDB: {self.stats['total_mongo_inserted']:,} inserted")
                self.logger.info(f"   Elasticsearch: {self.stats['total_es_indexed']:,} indexed")
                self.logger.info(f"   Elapsed: {elapsed_time:.1f} seconds")
            
    def _print_summary(self) -> None:
        """Print final summary."""
        elapsed_time = time.time() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("📊 DATA LOADING SUMMARY")
        print("="*60)
        print(f"Total Processing Time: {elapsed_time:.1f} seconds")
        print(f"Records Processed: {self.stats['total_processed']:,}")
        
        if self.validator:
            print(f"Records Validated: {self.stats['total_validated']:,}")
            print(f"Validation Errors: {self.stats['validation_errors']:,}")
            validation_rate = (self.stats['total_validated'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
            print(f"Validation Success Rate: {validation_rate:.1f}%")
        
        if self.mongo_handler:
            print(f"MongoDB Records Inserted: {self.stats['total_mongo_inserted']:,}")
            print(f"MongoDB Errors: {self.stats['mongo_errors']:,}")
            mongo_success_rate = (self.stats['total_mongo_inserted'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
            print(f"MongoDB Success Rate: {mongo_success_rate:.1f}%")
        
        if self.es_handler:
            print(f"Elasticsearch Records Indexed: {self.stats['total_es_indexed']:,}")
            print(f"Elasticsearch Errors: {self.stats['es_errors']:,}")
            es_success_rate = (self.stats['total_es_indexed'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
            print(f"Elasticsearch Success Rate: {es_success_rate:.1f}%")
        
        avg_rate = self.stats['total_processed'] / elapsed_time if elapsed_time > 0 else 0
        print(f"Average Processing Rate: {avg_rate:.1f} records/second")
        print("="*60)
    
    def close(self) -> None:
        """Close all connections."""
        if self.extractor:
            self.extractor.close()
        if self.mongo_handler:
            self.mongo_handler.close()
        if self.es_handler:
            self.es_handler.close()

        def main():
            """Main function."""
            args = parse_arguments()
            
            # Setup logging
            setup_logging(level=args.log_level)
            logger = logging.getLogger(__name__)
            
            logger.info("🚀 Starting Chicago 311 data loading...")
            
            # Validate arguments
            if not args.sample and not args.full:
                logger.error("❌ Please specify --sample or --full")
                sys.exit(1)
            
            if args.sample and args.full:
                logger.error("❌ Please specify either --sample or --full, not both")
                sys.exit(1)
            
            # Initialize data loader
            try:
                loader = DataLoader(args)
                
                # Load data based on arguments
                if args.sample:
                    loader.load_sample_data()
                elif args.full:
                    loader.load_full_data()
                
                # Close connections
                loader.close()
                
                logger.info("✅ Data loading completed successfully!")
                
            except KeyboardInterrupt:
                logger.warning("⚠️ Data loading interrupted by user")
                sys.exit(1)
            except Exception as e:
                logger.error(f"❌ Data loading failed: {e}")
                sys.exit(1)

        if __name__ == "__main__":
            main()