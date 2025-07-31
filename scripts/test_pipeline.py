#!/usr/bin/env python3
"""
Test script for Chicago 311 Data Pipeline
Validates all components and runs a small test
"""
import sys
import logging
import time

# Add src to path
sys.path.append('/Users/rahulmishra/Desktop/MS-Data_Science/Group Project/chicago-311-data-platform')

from src.data_extraction.chicago_data_extractor import validate_api_connection
from src.databases.mongodb_handler import test_mongodb_connection
from src.databases.elasticsearch_handler import test_elasticsearch_connection
from data_pipeline import ChicagoDataPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_connections():
    """Test all connections."""
    print("\n🧪 TESTING ALL CONNECTIONS")
    print("="*50)
    
    # Test API connection
    print("\n1. Testing Chicago Data Portal API...")
    api_success = validate_api_connection()
    print(f"   Result: {'✅ SUCCESS' if api_success else '❌ FAILED'}")
    
    # Test MongoDB connection
    print("\n2. Testing MongoDB connection...")
    mongo_success = test_mongodb_connection()
    print(f"   Result: {'✅ SUCCESS' if mongo_success else '❌ FAILED'}")
    
    # Test Elasticsearch connection
    print("\n3. Testing Elasticsearch connection...")
    es_success = test_elasticsearch_connection()
    print(f"   Result: {'✅ SUCCESS' if es_success else '❌ FAILED'}")
    
    all_success = api_success and mongo_success and es_success
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    return all_success

def run_mini_test():
    """Run a very small test with 10 records."""
    print("\n🔬 RUNNING MINI PIPELINE TEST (10 records)")
    print("="*50)
    
    try:
        pipeline = ChicagoDataPipeline()
        
        if not pipeline.initialize_connections():
            print("❌ Failed to initialize connections")
            return False
        
        # Run sample with just 10 records
        stats = pipeline.run_sample_pipeline(sample_size=10)
        
        success = (stats['total_extracted'] > 0 and 
                  stats['mongo_inserted'] > 0 and 
                  stats['es_indexed'] > 0)
        
        print(f"\n🎯 MINI TEST RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        pipeline.cleanup()
        return success
        
    except Exception as e:
        print(f"❌ Mini test error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Chicago 311 Data Pipeline Test Suite")
    print("="*60)
    
    # Test connections first
    if not test_all_connections():
        print("\n❌ Connection tests failed. Please check your configuration.")
        return 1
    
    # Run mini test
    if not run_mini_test():
        print("\n❌ Mini pipeline test failed.")
        return 1
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n📚 Usage Examples:")
    print("  # Sample run (1000 records):")
    print("  python scripts/data_pipeline.py --mode sample --sample-size 1000")
    print("\n  # Full run (all data):")
    print("  python scripts/data_pipeline.py --mode full")
    print("\n  # Incremental run (date range):")
    print("  python scripts/data_pipeline.py --mode incremental --start-date 2024-01-01 --end-date 2024-01-31")
    print("\n  # Show database statistics:")
    print("  python scripts/data_pipeline.py --stats-only")
    
    return 0

if __name__ == "__main__":
    exit(main())